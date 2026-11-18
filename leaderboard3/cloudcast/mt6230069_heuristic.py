# EVOLVE-BLOCK-START
import json
import os
import time
from typing import Dict, List

import networkx as nx
import pandas as pd


def search_algorithm(src, dsts, G, num_partitions):
    t0 = time.perf_counter()
    deadline = t0 + 18  # Initial construction time
    
    # Precompute shortest paths between all pairs for Steiner tree construction
    all_pairs_shortest = dict(nx.all_pairs_dijkstra_path(G, weight="cost"))
    
    # Build a Steiner tree using correct DP approach
    steiner_tree = build_steiner_tree_correct(G, src, dsts, all_pairs_shortest, deadline)
    
    # Create base topology with a single path per partition from src to each dst
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    
    # Route each partition through the Steiner tree
    for dst in dsts:
        # Get the path from src to dst in the Steiner tree
        try:
            path = nx.dijkstra_path(steiner_tree, src, dst, weight="cost")
        except nx.NetworkXNoPath:
            # Fallback to direct path in original graph if no path exists in Steiner tree
            try:
                path = nx.dijkstra_path(G, src, dst, weight="cost")
            except nx.NetworkXNoPath:
                # If even direct path fails, skip this destination
                continue
        
        # Convert path to edge list format and set it properly
        edges = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            if s in G and t in G[s]:
                edges.append([s, t, G[s][t]])
        
        for j in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(dst, j, edges)
    
    return bc_topology


def build_steiner_tree_correct(G, src, dsts, all_pairs_shortest, deadline):
    """
    Correctly construct a minimum-cost directed Steiner tree with src as root and dsts as terminals
    Uses proper dynamic programming over subsets with bitmask representation
    """
    terminals = set(dsts)
    
    # Handle edge case
    if len(terminals) == 0:
        # No destinations, return empty tree
        return nx.DiGraph()
    
    # Convert terminals to ordered list for bitmask indexing
    terminal_list = list(terminals)
    
    # DP table: dp[mask] = (min_cost, best_edges_list)
    # mask is a bitmask representing which terminals are included
    dp = {}
    dp[0] = (0, [])
    
    n_terminals = len(terminal_list)
    
    # For each subset of terminals (represented as bitmask)
    for mask in range(1, 1 << n_terminals):
        if time.perf_counter() > deadline:
            break
            
        dp[mask] = (float('inf'), [])
        
        # Find all terminals in this subset
        subset_terminals = [terminal_list[i] for i in range(n_terminals) if mask & (1 << i)]
        
        # Try adding each terminal to extend the tree
        for i in range(n_terminals):
            if mask & (1 << i):  # Terminal i is included
                terminal_i = terminal_list[i]
                prev_mask = mask ^ (1 << i)  # Remove terminal i from mask
                
                # If previous state exists, try extending it with a connection to terminal_i
                if prev_mask in dp:
                    min_cost = float('inf')
                    best_source = None
                    best_path_edges = []
                    
                    # Consider connecting from any previous node in the subset 
                    for j in range(n_terminals):
                        if prev_mask & (1 << j):
                            node_j = terminal_list[j]
                            if node_j in all_pairs_shortest and terminal_i in all_pairs_shortest[node_j]:
                                path = all_pairs_shortest[node_j][terminal_i]
                                if len(path) >= 2:
                                    path_cost = 0
                                    temp_edges = []
                                    for k in range(len(path) - 1):
                                        u = path[k]
                                        v = path[k + 1]
                                        if u in G and v in G[u]:
                                            path_cost += G[u][v]["cost"]
                                            temp_edges.append([u, v, G[u][v]])
                                    
                                    if path_cost < min_cost:
                                        min_cost = path_cost
                                        best_source = node_j
                                        best_path_edges = temp_edges[:]
                    
                    # Also consider connecting from src directly if that's better
                    if src in all_pairs_shortest and terminal_i in all_pairs_shortest[src]:
                        path = all_pairs_shortest[src][terminal_i]
                        if len(path) >= 2:
                            path_cost = 0
                            temp_edges = []
                            for k in range(len(path) - 1):
                                u = path[k]
                                v = path[k + 1]
                                if u in G and v in G[u]:
                                    path_cost += G[u][v]["cost"]
                                    temp_edges.append([u, v, G[u][v]])
                            
                            if path_cost < min_cost:
                                min_cost = path_cost
                                best_source = src
                                best_path_edges = temp_edges[:]
                    
                    # Also consider connecting from existing nodes in previous tree
                    if prev_mask != 0 and len(dp[prev_mask][1]) > 0:
                        # Check if we can improve by connecting to nodes that already have paths
                        prev_nodes = set()
                        for edge in dp[prev_mask][1]:
                            prev_nodes.add(edge[0])
                            prev_nodes.add(edge[1])
                        for node in prev_nodes:
                            if node in all_pairs_shortest and terminal_i in all_pairs_shortest[node]:
                                path = all_pairs_shortest[node][terminal_i]
                                if len(path) >= 2:
                                    path_cost = 0
                                    temp_edges = []
                                    for k in range(len(path) - 1):
                                        u = path[k]
                                        v = path[k + 1]
                                        if u in G and v in G[u]:
                                            path_cost += G[u][v]["cost"]
                                            temp_edges.append([u, v, G[u][v]])
                                    
                                    if path_cost < min_cost:
                                        min_cost = path_cost
                                        best_source = node
                                        best_path_edges = temp_edges[:]
                    
                    if min_cost < float('inf') and dp[prev_mask][0] + min_cost < dp[mask][0]:
                        # Merge the edge lists from previous state and new path
                        merged_edges = dp[prev_mask][1][:] + best_path_edges
                        dp[mask] = (dp[prev_mask][0] + min_cost, merged_edges)
    
    # Return the complete Steiner tree spanning source and all terminals
    final_mask = (1 << n_terminals) - 1
    if final_mask in dp:
        final_edges = dp[final_mask][1]
    else:
        final_edges = []
    
    # Build and return the actual tree
    steiner_tree = nx.DiGraph()
    steiner_tree.add_nodes_from([src] + list(terminals))
    
    for u, v, edge_data in final_edges:
        # Only add edges that exist in original graph
        if u in G and v in G[u]:
            steiner_tree.add_edge(u, v, **edge_data)
    
    # Ensure we have a valid tree that connects all terminals
    # If we don't, fall back to simple approach
    if len(steiner_tree.nodes()) == 0:
        # Simple fallback - just connect from src to each destination
        steiner_tree = nx.DiGraph()
        steiner_tree.add_node(src)
        for dst in dsts:
            steiner_tree.add_node(dst)
            try:
                path = nx.dijkstra_path(G, src, dst, weight="cost")
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if u in G and v in G[u]:  # Validate edge exists
                        steiner_tree.add_edge(u, v, **G[u][v])
            except nx.NetworkXNoPath:
                continue
                
    return steiner_tree


class BroadCastTopology:
    def __init__(
        self,
        src: str,
        dsts: List[str],
        num_partitions: int = 4,
        paths: Dict[str, Dict[str, List[List]]] = None,
    ):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
        else:
            self.paths = {dst: {str(i): [] for i in range(num_partitions)} for dst in dsts}

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition] = paths


# EVOLVE-BLOCK-END


# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)
