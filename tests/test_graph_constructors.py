import pytest
import ase.io
import networkx as nx
from tmc_tools.graphs.constructors import graph_from_ase_atoms, graph_from_mol_file


def test_water(resource_path_root):
    g_ref = nx.Graph()
    g_ref.add_nodes_from(
        [(0, {"atomic_number": 8}), (1, {"atomic_number": 1}), (2, {"atomic_number": 1})]
    )
    g_ref.add_edges_from([(0, 1), (0, 2)])

    atoms = ase.io.read(resource_path_root / "water.mol")
    g = graph_from_ase_atoms(atoms)
    assert g.nodes == g_ref.nodes
    assert g.edges == g_ref.edges

    g = graph_from_mol_file(resource_path_root / "water.mol")
    assert g.nodes == g_ref.nodes
    assert g.edges == g_ref.edges


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


def test_furan(resource_path_root, furan_graph):

    atoms = ase.io.read(resource_path_root / "furan.mol")
    g = graph_from_ase_atoms(atoms)
    assert g.nodes == furan_graph.nodes
    assert g.edges == furan_graph.edges

    g = graph_from_mol_file(resource_path_root / "furan.mol")
    assert g.nodes == furan_graph.nodes
    assert g.edges == furan_graph.edges
