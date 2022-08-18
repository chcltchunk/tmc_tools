import pytest
import numpy as np
import networkx as nx
import operator
from tmc_tools.graphs.racs import atom_centered_AC, multi_centered_AC


@pytest.fixture
def furan_graph():
    g = nx.Graph()
    g.add_nodes_from(
        [
            (0, {"atomic_number": 8}),
            (1, {"atomic_number": 6}),
            (2, {"atomic_number": 6}),
            (3, {"atomic_number": 6}),
            (4, {"atomic_number": 6}),
            (5, {"atomic_number": 1}),
            (6, {"atomic_number": 1}),
            (7, {"atomic_number": 1}),
            (8, {"atomic_number": 1}),
        ]
    )
    g.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 5), (2, 6), (4, 7), (3, 8)]
    )
    return g


@pytest.fixture
def fe_co_6():
    g = nx.Graph()
    g.add_node(0, atomic_number="Fe")
    for i in range(6):
        g.add_node(1 + 2 * i, atomic_number="C")
        g.add_edge(0, 1 + 2 * i)
        g.add_node(2 + 2 * i, atomic_number="O")
        g.add_edge(1 + 2 * i, 2 + 2 * i)
    return g


def test_atom_centered_AC(furan_graph):
    descriptors = atom_centered_AC(furan_graph, 0, depth=3)
    # properties: Z, chi, T,  I, S
    ref = [
        [64.0, 11.8336, 4.0, 1.0, 0.4356],
        [96.0, 17.544, 12.0, 2.0, 1.0032],
        [112.0, 32.68, 16.0, 4.0, 1.4124],
        [16.0, 15.136, 4.0, 2.0, 0.4092],
    ]
    np.testing.assert_allclose(descriptors, ref)


def test_atom_centered_AC_diff(furan_graph):
    descriptors = atom_centered_AC(furan_graph, 0, depth=3, operation=operator.sub)
    # properties: Z, chi, T,  I, S
    ref = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 1.78, -2.0, 0.0, -0.2],
        [18.0, 4.26, 0.0, 0.0, 0.5],
        [14.0, 2.48, 2.0, 0.0, 0.7],
    ]
    np.testing.assert_allclose(descriptors, ref)


def test_multi_centered_AC(furan_graph):
    descriptors = multi_centered_AC(furan_graph, depth=3)
    # properties: Z, chi, T,  I, S
    ref = [
        [212.0, 57.2036, 44.0, 9.0, 3.1304],
        [456.0, 118.983, 102.0, 18.0, 7.3568],
        [512.0, 171.695, 122.0, 26.0, 9.1176],
        [110.0, 126.632, 50.0, 22.0, 4.2222],
    ]
    np.testing.assert_allclose(descriptors, ref)
