// Copyright 2026 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Generic implementation of the "closest common dominator" algorithm for
//! directed graphs.
//!
//! Generic implementation of the Common Dominator algorithm for directed
//! graphs, using the Cooper-Harvey-Kennedy iterative algorithm. Loosely
//! speaking the algorithm finds the "choke point" for a set of nodes S in a
//! directed graph (going from the "entry" node to nodes in S), closest to S.
//!
//! Dominance:
//!
//! * A flow graph is a directed graph with a designated entry node.
//! * A node z is said to dominate a node n if all paths from the entry node to
//!   n must go through z. Every node dominates itself, and the entry node
//!   dominates all nodes.
//! * A node can have one or more dominators.
//! * A node z strictly dominates n if z dominates n and z != n.
//! * The immediate dominator of a node n is the dominator of n that doesn't
//!   strictly dominate any other strict dominators of n. Informally it is the
//!   "closest" choke point on all paths from the entry node to n.
//! * Let S be a subset of the nodes in the graph. The intersection of the
//!   dominators of each node in S is the set of common dominators of S.
//! * The closest common dominator of S is the common dominator of S that
//!   doesn't strictly dominate any other common dominator of S. Informally, it
//!   is the choke point closest to S such that all paths from the entry node to
//!   S must go through it.
//!
//! Dominator Tree:
//!
//! For any flow graph G there is a corresponding dominator tree defined as
//! follows:
//! * The nodes of the dominator tree are the same as the nodes of G
//! * The root of the dominator tree is the entry node of G
//! * In the dominator tree, the children of a node are the nodes it immediately
//!   dominates
//!
//! The closest common dominator of S is the Lowest Common Ancestor (LCA)
//! of S in the graph's dominator tree.
//!
//! This implementation constructs the Dominator Tree by first determining
//! the Immediate Dominator for every node (using the standard iterative
//! algorithm), and then calculating the LCA for the set S. See:
//!
//! * <http://www.hipersoft.rice.edu/grads/publications/dom14.pdf>
//! * <https://en.wikipedia.org/wiki/Dominator_(graph_theory)>
//!
//! The running time is O(V+E+|S|*V)in the worst case, the space complexity is
//! O(V+E), where V is the number of nodes and E is the number of edges. In
//! practice the algorithm is fast and efficient for typical use cases because
//! the number of nodes that dominate any given node is typically small, and
//! the dominator tree is typically shallow.

use std::collections::HashMap;
use std::hash::Hash;

use indexmap::IndexMap;
use indexmap::IndexSet;
use itertools::Itertools as _;
use jj_lib::dag_walk::post_order;

/// An immutable directed graph with nodes of type N and a minimal interface for
/// iterating over nodes and their adjacent nodes.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct SimpleDirectedGraph<N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// The adjacency map of the graph. Each key is a node, and the
    /// corresponding value is the set of adjacent nodes (i.e., the children of
    /// the key node). The adjacency map is in canonical form: for every
    /// u->v edge, there is an entry in adj with key v (even if v has no
    /// outgoing edges).
    adj: IndexMap<N, IndexSet<N>>,
}

impl<N> SimpleDirectedGraph<N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// Constructs a new SimpleDirectedGraph from an adjacency map.
    pub fn new(mut adj: IndexMap<N, IndexSet<N>>) -> Self {
        let mut missing_nodes = IndexSet::new();
        for (_, children) in &adj {
            for child in children {
                if !adj.contains_key(child) {
                    missing_nodes.insert(child.clone());
                }
            }
        }
        for node in missing_nodes {
            adj.entry(node).or_default();
        }
        Self { adj }
    }

    /// Constructs a new graph from a list of edges. Iteration order is
    /// preserved from the input.
    pub fn from_edge_list<EI>(edges: EI) -> Self
    where
        EI: IntoIterator<Item = (N, N)>,
    {
        let mut adj: IndexMap<N, IndexSet<N>> = IndexMap::new();
        for (parent, child) in edges {
            adj.entry(parent).or_default().insert(child);
        }
        Self::new(adj)
    }

    /// Returns the nodes in this graph.
    pub fn nodes(&self) -> impl Iterator<Item = &N> {
        self.adj.keys()
    }

    /// Returns the nodes in this graph.
    pub fn num_nodes(&self) -> usize {
        self.adj.len()
    }

    /// Returns the edges in this graph.
    pub fn edges(&self) -> impl Iterator<Item = (&N, &N)> {
        self.adj
            .iter()
            .flat_map(|(parent, adj_set)| adj_set.iter().map(move |child| (parent, child)))
    }

    /// Returns the adjacent nodes for the given node, or None if the node is
    /// not in the graph.
    pub fn adjacent_nodes(&self, node: &N) -> Option<impl Iterator<Item = &N>> {
        self.adj.get(node).map(|adj_set| adj_set.iter())
    }

    /// Returns true if this graph contains the given node.
    pub fn contains_node(&self, node: &N) -> bool {
        self.adj.contains_key(node)
    }

    /// Returns a postorder traversal of the nodes in this graph starting from
    /// the given node.
    pub fn get_postorder<'a>(&'a self, start_node: &'a N) -> Vec<&'a N> {
        post_order(start_node, |&u| {
            self.adjacent_nodes(u).into_iter().flatten()
        })
        .collect_vec()
    }

    /// Returns a new graph with the same nodes as this graph and all edges
    /// reversed.
    pub fn reverse(&self) -> Self {
        let mut rev_adj: IndexMap<N, IndexSet<N>> = IndexMap::new();
        for (parent, children) in &self.adj {
            // Ensure parent is in rev_adj even if it has no children.
            rev_adj.entry(parent.clone()).or_default();
            for child in children {
                rev_adj
                    .entry(child.clone())
                    .or_default()
                    .insert(parent.clone());
            }
        }
        Self { adj: rev_adj }
    }
}

/// A FlowGraph is a directed graph with a designated start node.
///
/// Any node in the graph can be the start node. There are no reachability
/// requirements whatsoever: some nodes may be unreachable from the start node,
/// the start node could have incoming edges, the graph could be disconnected,
/// etc.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct FlowGraph<N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// The graph.
    pub graph: SimpleDirectedGraph<N>,
    /// The start node.
    pub start_node: N,
}

impl<N> FlowGraph<N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// Constructs a new FlowGraph.
    pub fn new(graph: SimpleDirectedGraph<N>, start_node: N) -> Self {
        Self { graph, start_node }
    }
}

/// Calculates the dominators in a flow graph. Also has a method for finding the
/// closest common dominator of a set of nodes.
pub struct DominatorFinder<'a, N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// The flow graph.
    pub flow_graph: &'a FlowGraph<N>,
    /// Map from nodes to integers in [0, N-1] range, in postorder (the start
    /// node has index N-1).
    node_to_id: HashMap<&'a N, Index>,
    /// The inverse of node_to_id.
    id_to_node: Vec<&'a N>,
    /// The immediate dominator for each node (by index). NOTE: the immediate
    /// dominator of the start node is itself.
    immediate_dominators: Vec<Option<Index>>,
}

/// Type alias for clarity.
type Index = usize;

impl<'a, N> DominatorFinder<'a, N>
where
    N: Clone + Eq + Hash + PartialEq,
{
    /// Constructs a new DominatorFinder. Returns an error if the flow graph is
    /// invalid: e.g. if some node is unreachable from the start node.
    pub fn calculate(flow_graph: &'a FlowGraph<N>) -> Result<Self, String> {
        // Get postorder traversal of the graph starting from the start node.
        let postorder = flow_graph.graph.get_postorder(&flow_graph.start_node);
        if postorder.len() != flow_graph.graph.num_nodes() {
            return Err(
                "Invalid flow graph: some nodes are unreachable from the start node".to_string(),
            );
        }

        // Map generic types to integer IDs
        let mut node_to_id = HashMap::new();
        let mut id_to_node = Vec::new();
        for (index, &node) in postorder.iter().enumerate() {
            id_to_node.push(node);
            node_to_id.insert(node, index);
        }

        // Build graph using internal IDs.
        let num_nodes = node_to_id.len();

        let mut adj = vec![vec![]; num_nodes];
        let mut rev_adj = vec![vec![]; num_nodes];
        for (u, v) in flow_graph.graph.edges() {
            if let (Some(u_idx), Some(v_idx)) = (node_to_id.get(u), node_to_id.get(v)) {
                adj[*u_idx].push(*v_idx);
                rev_adj[*v_idx].push(*u_idx);
            }
        }

        // Find the immediate dominators for each node using the Cooper-Harvey-Kennedy
        // iterative algorithm.
        let immediate_dominators = Self::get_immediate_dominators_internal(&adj, &rev_adj);

        Ok(Self {
            flow_graph,
            node_to_id,
            id_to_node,
            immediate_dominators,
        })
    }

    /// Returns a map from each node to its immediate dominator. NOTE: the
    /// immediate dominator of the start node is itself.
    pub fn get_immediate_dominators(&self) -> HashMap<N, N> {
        self.immediate_dominators
            .iter()
            .enumerate()
            .map(|(index, idom_opt)| {
                let idom = idom_opt.unwrap();
                (
                    self.id_to_node[index].clone(),
                    self.id_to_node[idom].clone(),
                )
            })
            .collect()
    }

    /// Finds the closest common dominator for the given flow graph and set of
    /// nodes S (target_set).
    pub fn find_closest_common_dominator<NI>(&self, target_set: NI) -> Result<N, String>
    where
        NI: IntoIterator<Item = N>,
    {
        // Convert generic target_set to internal IDs
        let target_ids: Option<Vec<Index>> = target_set
            .into_iter()
            .map(|node| self.node_to_id.get(&node).copied())
            .collect();
        let target_ids = if target_ids.as_ref().is_none_or(|ids| ids.is_empty()) {
            return Err(
                "Target set is empty or has nodes not present in the flow graph".to_string(),
            );
        } else {
            target_ids.unwrap()
        };

        // The closest common dominator of a set of nodes is the lowest common ancestor
        // of those nodes in the dominator tree.
        let closest_common_dominator =
            Self::find_lowest_common_ancestor(&target_ids, &self.immediate_dominators);

        // Map the internal ID back to generic type N.
        Ok(self.id_to_node[closest_common_dominator].clone())
    }

    // Applies the Cooper-Harvey-Kennedy iterative algorithm to find the immediate
    // dominators for each node in the graph.
    // See http://www.hipersoft.rice.edu/grads/publications/dom14.pdf for details on how this function works.
    fn get_immediate_dominators_internal(
        adj: &[Vec<Index>],
        rev_adj: &[Vec<Index>],
    ) -> Vec<Option<Index>> {
        // Step 1: Compute Dominators on Reverse Graph
        let num_nodes = adj.len();
        let start_node_id = num_nodes - 1;

        // We hold the immediate dominator for each node in the following vector, in
        // index position (the kth entry is the immediate dominator of the node with ID
        // k). We initialize the immediate dominator of every node to None, except for
        // the start node which is initialized to itself. The None value means "not
        // processed yet". Once a node is processed, its immediate dominator is
        // guaranteed to be Some.
        let mut immediate_dominators: Vec<Option<Index>> = vec![None; num_nodes];
        // NOTE: technically speaking the immediate dominator is NOT defined for the start node, but it is convenient for the algorithm to set it to itself; this is consistent with the literature and specifically with http://www.hipersoft.rice.edu/grads/publications/dom14.pdf
        immediate_dominators[start_node_id] = Some(start_node_id);

        loop {
            // Each iteration of the loop processes all nodes in reverse postorder, trying
            // to improve the immediate dominator for each node. The loop continues until we
            // have an iteration where no immediate dominator is changed. Note that the
            // entries in immediate_dominators are only guaranteed to be correct when the
            // loop terminates.
            let mut changed = false;

            // Iterate in reverse postorder, skipping the start node.
            for u in (0..start_node_id).rev() {
                let mut new_idom = None;

                // Process predecessors (nodes that flow INTO u).
                let preds = &rev_adj[u];

                // Find first processed predecessor.
                for &p in preds {
                    if immediate_dominators[p].is_some() {
                        new_idom = Some(p);
                        break;
                    }
                }

                if let Some(mut candidate) = new_idom {
                    for &p in preds {
                        if p != candidate && immediate_dominators[p].is_some() {
                            candidate = Self::intersect(candidate, p, &immediate_dominators);
                        }
                    }

                    if immediate_dominators[u] != Some(candidate) {
                        immediate_dominators[u] = Some(candidate);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // At this point we know the immediate dominator of every node, but we keep the
        // Option wrapper so that we can use the intersect function during
        // find_lowest_common_ancestor.
        immediate_dominators
    }

    // See http://www.hipersoft.rice.edu/grads/publications/dom14.pdf for details on how this function works.
    fn intersect(mut b1: Index, mut b2: Index, immediate_dominators: &[Option<Index>]) -> Index {
        while b1 != b2 {
            while b1 < b2 {
                b1 = immediate_dominators[b1].unwrap();
            }
            while b2 < b1 {
                b2 = immediate_dominators[b2].unwrap();
            }
        }
        b1
    }

    // See http://www.hipersoft.rice.edu/grads/publications/dom14.pdf for details on how this function works.
    fn find_lowest_common_ancestor(
        targets: &[Index],
        immediate_dominators: &[Option<Index>],
    ) -> Index {
        assert!(!targets.is_empty());
        let mut lca = targets[0];
        for &node in &targets[1..] {
            lca = Self::intersect(lca, node, immediate_dominators);
        }
        lca
    }
}

#[cfg(test)]
mod tests {
    use indexmap::indexmap;
    use indexmap::indexset;

    use super::*;

    #[test]
    fn test_closest_common_dominator_split() -> Result<(), String> {
        //   /-> B \
        // A        -> D
        //   \-> C /
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["A"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["D"])?, "D");
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "D"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "C", "D"])?, "A");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_linear_chain() -> Result<(), String> {
        // A -> B -> C -> D
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([("A", "B"), ("B", "C"), ("C", "D")]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["A"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["D"])?, "D");
        assert_eq!(df.find_closest_common_dominator(["A", "B"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["A", "C"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["A", "D"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "D"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C", "D"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["A", "B", "C", "D"])?, "A");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_classic_diamond() -> Result<(), String> {
        //      /-> B -\
        //    A          -> D -> E
        //      \-> C -/
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "D"),
                ("D", "E"),
            ]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "E"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["D"])?, "D");
        assert_eq!(df.find_closest_common_dominator(["D", "E"])?, "D");
        assert_eq!(df.find_closest_common_dominator(["A", "D"])?, "A");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_single_node() -> Result<(), String> {
        // A
        let flow_graph = FlowGraph::new(SimpleDirectedGraph::from_edge_list([("A", "A")]), "A");
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["A"])?, "A");
        Ok(())
    }

    #[test]
    fn test_invalid_flowgraph() {
        //       /-> E
        // A -> B
        //       \-> F
        //           ^
        //           |
        // C --> D --/
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([
                ("A", "B"),
                ("B", "E"),
                ("B", "F"),
                ("C", "D"),
                ("D", "F"),
            ]),
            "A",
        );
        assert_eq!(
            DominatorFinder::calculate(&flow_graph).err(),
            Some("Invalid flow graph: some nodes are unreachable from the start node".to_string())
        );
    }

    #[test]
    fn test_closest_common_dominator_simple_cycle_with_entry() -> Result<(), String> {
        //
        // A -> B -> C -> D
        //      ^         |
        //      |         |
        //      \--------/
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["A", "B"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["A", "C"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["A", "B", "C"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "C", "D"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["A"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["D"])?, "D");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_figure_eight_with_bridge() -> Result<(), String> {
        //
        //  A -> B -> C -> D -> E -> F -> G
        //       ^         |    ^         |
        //       |         |    |         |
        //        \_______/      \_______/
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([
                ("A", "B"), // entry
                ("B", "C"),
                ("C", "D"),
                ("D", "B"), // Loop 1
                ("D", "E"), // Bridge
                ("E", "F"),
                ("F", "G"),
                ("G", "E"), // Loop 2
            ]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "D"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "E"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C", "E"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["C", "F"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["D", "E"])?, "D");
        assert_eq!(df.find_closest_common_dominator(["D", "F"])?, "D");
        assert_eq!(df.find_closest_common_dominator(["E", "G"])?, "E");
        assert_eq!(df.find_closest_common_dominator(["F", "G"])?, "F");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_figure_eight() -> Result<(), String> {
        //
        //  A -> B -> C --> D   -> E -> F
        //       ^         | ^          |
        //       |         | |          |
        //        \_______/  \_________/
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([
                ("A", "B"), // entry
                ("B", "C"),
                ("C", "D"),
                ("D", "B"), // Loop 1
                ("D", "E"),
                ("E", "F"),
                ("F", "D"), // Loop 2
            ]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "D"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "E"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C", "D"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["C", "E"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["C", "F"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["D", "E"])?, "D");
        assert_eq!(df.find_closest_common_dominator(["D", "F"])?, "D");
        assert_eq!(df.find_closest_common_dominator(["E", "F"])?, "E");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_entry_cycle_dominance() -> Result<(), String> {
        // A -> B -> C
        //      ^    |
        //      |----/
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([("A", "B"), ("B", "C"), ("C", "B")]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["A", "B"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["A", "C"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["A", "B", "C"])?, "A");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_nested_loops() -> Result<(), String> {
        //           /---> E
        //           |     |
        //           |     |
        // A -> B -> C <--/
        //      ^    |
        //      |    V
        //      \----D
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("C", "E"),
                ("E", "C"),
                ("D", "B"),
            ]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["A", "B"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["A", "C"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "D"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "E"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C", "D"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["C", "E"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["D", "E"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["B", "C", "D"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "C", "E"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "D", "E"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C", "D", "E"])?, "C");
        assert_eq!(df.find_closest_common_dominator(["B", "C", "D", "E"])?, "B");
        Ok(())
    }

    #[test]
    fn test_irreducible_graph_cooper_harvey_kennedy_fig2() -> Result<(), String> {
        //        5
        //     /    \
        //    |      |
        //    V      V
        //    4      3
        //    |      |
        //    V      V
        //    1 <==> 2
        let graph =
            SimpleDirectedGraph::from_edge_list([(1, 2), (2, 1), (3, 2), (4, 1), (5, 4), (5, 3)]);
        let flow_graph = FlowGraph::new(graph, 5);
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(
            df.get_immediate_dominators(),
            HashMap::from([(1, 5), (2, 5), (3, 5), (4, 5), (5, 5),])
        );
        Ok(())
    }

    #[test]
    fn test_irreducible_graph_cooper_harvey_kennedy_fig3() -> Result<(), String> {
        //     6
        //   /   \
        //  |     |
        //  v     v
        //  5     4 --
        //  |     |    \
        //  v     v     v
        //  1 <=> 2 <=> 3
        let graph = SimpleDirectedGraph::from_edge_list([
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (5, 1),
            (4, 2),
            (4, 3),
            (6, 5),
            (6, 4),
        ]);
        let flow_graph = FlowGraph::new(graph, 6);
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(
            df.get_immediate_dominators(),
            HashMap::from([(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6),])
        );
        assert_eq!(df.find_closest_common_dominator([2, 3])?, 6);
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_tree() -> Result<(), String> {
        // A -> B -> C
        // \     \-> D
        //  \------> E
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([("A", "B"), ("B", "C"), ("B", "D"), ("A", "E")]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "E"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["C", "D"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C", "E"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "C", "D"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["C", "D", "E"])?, "A");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_bypassing_path() -> Result<(), String> {
        // A -> B -> C -> D
        // |              ^
        // v              |
        // E -------------/
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("A", "E"),
                ("E", "D"),
            ]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "B");
        assert_eq!(df.find_closest_common_dominator(["B", "D"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "E"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["C", "D"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["C", "E"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["D", "E"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "C", "D"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["C", "D", "E"])?, "A");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_self_loop_handling() -> Result<(), String> {
        // A->A (Self loop), A->B
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([("A", "A"), ("A", "B")]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["A"])?, "A");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_multi_edge() -> Result<(), String> {
        // Shape: A->B (x2), B->C.
        let flow_graph = FlowGraph::new(
            SimpleDirectedGraph::from_edge_list([
                ("A", "B"),
                ("A", "B"), // Duplicate edge
                ("B", "C"),
            ]),
            "A",
        );
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["A"])?, "A");
        assert_eq!(df.find_closest_common_dominator(["B", "C"])?, "B");
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_invalid_target_set() -> Result<(), String> {
        // A -> B
        let flow_graph = FlowGraph::new(SimpleDirectedGraph::from_edge_list([("A", "B")]), "A");
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(
            df.find_closest_common_dominator([]),
            Err("Target set is empty or has nodes not present in the flow graph".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_closest_common_dominator_repeated_node() -> Result<(), String> {
        // A -> B
        let flow_graph = FlowGraph::new(SimpleDirectedGraph::from_edge_list([("A", "B")]), "A");
        let df = DominatorFinder::calculate(&flow_graph)?;
        assert_eq!(df.find_closest_common_dominator(["A", "B", "A", "B"])?, "A");
        Ok(())
    }

    #[test]
    fn test_simple_directed_graph_new() {
        let adj = indexmap! {
            "A" => indexset! {"B"},
            "B" => indexset!{},
        };
        let graph = SimpleDirectedGraph::new(adj.clone());
        assert_eq!(graph.adj, adj);

        // adj does not have entries for "B" or "D".
        let adj = indexmap! {
            "A" => indexset! {"B", "C", "D"},
            "C" => indexset!{},
        };
        let graph = SimpleDirectedGraph::new(adj);
        assert_eq!(
            graph.adj,
            indexmap! {
                "A" => indexset! {"B", "C", "D"},
                "C" => indexset!{},
                "B" => indexset!{},
                "D" => indexset!{},
            }
        );
    }

    #[test]
    fn test_simple_directed_graph_nodes() {
        let graph = SimpleDirectedGraph::from_edge_list([("A", "B"), ("B", "C")]);
        let nodes = graph.nodes().copied().collect_vec();
        assert_eq!(nodes, ["A", "B", "C"]);

        let graph = SimpleDirectedGraph::<String>::from_edge_list([]);
        let nodes = graph.nodes().cloned().collect_vec();
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_simple_directed_graph_edges() {
        let graph = SimpleDirectedGraph::from_edge_list([("A", "B"), ("B", "C"), ("A", "C")]);
        let edges = graph.edges().map(|(&u, &v)| (u, v)).collect_vec();
        assert_eq!(edges, [("A", "B"), ("A", "C"), ("B", "C")]);

        let graph = SimpleDirectedGraph::<String>::from_edge_list([]);
        let edges = graph.edges().collect_vec();
        assert!(edges.is_empty());
    }

    #[test]
    fn test_simple_directed_graph_adjacent_nodes() {
        let graph = SimpleDirectedGraph::from_edge_list([("A", "B"), ("A", "C"), ("B", "D")]);
        assert_eq!(
            graph.adjacent_nodes(&"A").unwrap().copied().collect_vec(),
            ["B", "C"]
        );
        assert_eq!(
            graph.adjacent_nodes(&"B").unwrap().copied().collect_vec(),
            ["D"]
        );
        assert!(graph.adjacent_nodes(&"C").unwrap().next().is_none());
        assert!(graph.adjacent_nodes(&"Z").is_none());
    }

    #[test]
    fn test_simple_directed_graph_contains_node() {
        let graph = SimpleDirectedGraph::from_edge_list([("A", "B"), ("B", "C")]);
        assert!(graph.contains_node(&"A"));
        assert!(graph.contains_node(&"B"));
        assert!(graph.contains_node(&"C"));
        assert!(!graph.contains_node(&"D"));
    }

    #[test]
    fn test_simple_directed_graph_from_edge_list() {
        let graph =
            SimpleDirectedGraph::from_edge_list([("A", "B"), ("A", "C"), ("B", "C"), ("A", "B")]);
        let nodes = graph.nodes().copied().collect_vec();
        assert_eq!(nodes, ["A", "B", "C"]);
        let edges = graph.edges().map(|(&u, &v)| (u, v)).collect_vec();
        assert_eq!(edges, [("A", "B"), ("A", "C"), ("B", "C")]);

        let graph = SimpleDirectedGraph::from_edge_list([("B", "C"), ("A", "B")]);
        let nodes = graph.nodes().copied().collect_vec();
        assert_eq!(nodes, ["B", "A", "C"]);
        let edges = graph.edges().map(|(&u, &v)| (u, v)).collect_vec();
        assert_eq!(edges, [("B", "C"), ("A", "B")]);
    }

    #[test]
    fn test_flow_graph_new() {
        let graph = SimpleDirectedGraph::from_edge_list([("A", "B")]);
        let flow_graph = FlowGraph::new(graph.clone(), "A");
        assert_eq!(flow_graph.graph, graph);
        assert_eq!(flow_graph.start_node, "A");
        let flow_graph = FlowGraph::new(graph.clone(), "C");
        assert_eq!(flow_graph.graph, graph);
        assert_eq!(flow_graph.start_node, "C");
    }
}
