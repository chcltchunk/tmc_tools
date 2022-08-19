import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import norm as spnorm
from tmc_tools.constants import metal_list 



def compute_graph_determinant(graph):
    # TODO: test
    # compute graph determinant 
    # according to https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.0c01458
    weights = diags(list(nx.get_node_attributes(graph, "atomic_number").values()))
    A = nx.adjacency_matrix(graph) 
    weighted_A = weights@A@weights
    return np.linalg.det(weighted_A.todense())

def compute_graph_infinity_norm(graph):
    # TODO: test
    # compute graph infinity norm
    # might replace determinant (#TODO) 
    weights = diags(list(nx.get_node_attributes(graph, "atomic_number").values()))
    A = nx.adjacency_matrix(graph) 
    weighted_A = weights@A@weights
    return spnorm(weighted_A, 'inf')

def get_metal_id(graph):
    for node in graph.nodes():
        if graph.nodes[node]['atomic_number'] in metal_list:
            return node
    return None

