import pytest
import numpy as np
import networkx as nx
import operator
from tmc_tools.graphs.racs import (atom_centered_AC,
                                   multi_centered_AC)


@pytest.fixture
def furan_graph():
    g = nx.Graph()
    g.add_nodes_from([(0, {'symbol': 'O'}),
                      (1, {'symbol': 'C'}),
                      (2, {'symbol': 'C'}),
                      (3, {'symbol': 'C'}),
                      (4, {'symbol': 'C'}),
                      (5, {'symbol': 'H'}),
                      (6, {'symbol': 'H'}),
                      (7, {'symbol': 'H'}),
                      (8, {'symbol': 'H'})])
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 5),
                      (2, 6), (4, 7), (3, 8)])
    return g


@pytest.fixture
def fe_co_6():
    g = nx.Graph()
    g.add_node(0, symbol='Fe')
    for i in range(6):
        g.add_node(1 + 2*i, symbol='C')
        g.add_edge(0, 1 + 2*i)
        g.add_node(2 + 2*i, symbol='O')
        g.add_edge(1 + 2*i, 2 + 2*i)
    return g


def test_atom_centered_AC(furan_graph):
    descriptors = atom_centered_AC(furan_graph, 0, depth=3)
    # properties: Z, chi, T,  I, S
    ref = [[64., 11.8336, 4., 1., 0.4356],
           [96., 17.544, 12., 2., 1.0032],
           [112., 32.68, 16., 4., 1.4124],
           [16., 15.136, 4., 2., 0.4092]]
    np.testing.assert_allclose(descriptors, ref)


def test_atom_centered_AC_diff(furan_graph):
    descriptors = atom_centered_AC(furan_graph, 0, depth=3,
                                   operation=operator.sub)
    # properties: Z, chi, T,  I, S
    ref = [[0., 0., 0., 0., 0.],
           [4., 1.78, -2., 0., -0.2],
           [18., 4.26, 0., 0., 0.5],
           [14., 2.48, 2., 0., 0.7]]
    np.testing.assert_allclose(descriptors, ref)


def test_multi_centered_AC(furan_graph):
    descriptors = multi_centered_AC(furan_graph, depth=3)
    # properties: Z, chi, T,  I, S
    ref = [[212., 57.2036, 44., 9., 3.1304],
           [456., 118.983, 102., 18., 7.3568],
           [512., 171.695, 122., 26., 9.1176],
           [110., 126.632, 50., 22., 4.2222]]
    np.testing.assert_allclose(descriptors, ref)
