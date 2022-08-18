import numpy as np
import networkx as nx
import operator
from tmc_tools.constants import atomic_numbers, electronegativity, covalent_radii


def racs_property_vector(graph, node):
    output = np.zeros(5)
    Z = graph.nodes[node]["atomic_number"]
    # property (i): nuclear charge Z
    output[0] = Z
    # property (ii): Pauling electronegativity chi
    output[1] = electronegativity[Z]
    # property (iii): topology T, coordination number
    output[2] = len(list(graph.neighbors(node)))
    # property (iv): identity
    output[3] = 1.0
    # property (v): covalent radius S
    output[4] = covalent_radii[Z]
    return output


def atom_centered_AC(
    graph,
    starting_node,
    depth: int = 3,
    operation=operator.mul,
    property_fun=racs_property_vector,
):
    # Generate all paths from the starting node to all possible nodes
    lengths = nx.single_source_shortest_path_length(
        graph, source=starting_node, cutoff=depth
    )
    p_i = property_fun(graph, starting_node)
    output = np.zeros((depth + 1, len(p_i)))
    for node, d_ij in lengths.items():
        p_j = property_fun(graph, node)
        output[d_ij] += operation(p_i, p_j)
    return output


def multi_centered_AC(
    graph, depth: int = 3, operation=operator.mul, property_fun=racs_property_vector
):
    n_props = len(property_fun(graph, list(graph.nodes.keys())[0]))
    output = np.zeros((depth + 1, n_props))
    # Generate all pairwise path lengths
    lengths = nx.all_pairs_shortest_path_length(graph, cutoff=depth)
    for node_i, lengths_i in lengths:
        p_i = property_fun(graph, node_i)
        for node_j, d_ij in lengths_i.items():
            p_j = property_fun(graph, node_j)
            output[d_ij] += operation(p_i, p_j)
    return output


def ocatahedral_racs(
    graph,
    depth: int = 3,
    equatorial_connecting_atoms=None,
    property_fun=racs_property_vector,
):
    # Following J. Phys. Chem. A 2017, 121, 8939 there are 6 start/scope
    # combinations for product ACs and 3 for difference ACs.
    n_props = len(property_fun(graph, list(graph.nodes.keys())[0]))
    output = np.zeros((6 + 3, depth + 1, n_props))

    # start = f, scope = all, product
    output[0] = multi_centered_AC(graph, depth=depth, property_fun=property_fun)
    # start = mc, scope = all, product
    output[1] = atom_centered_AC(graph, 0, depth=depth, property_fun=property_fun)

    # For the other scopes the graph has to be subdivided into individual
    # ligand graphs. Make these changes on a copy of the graph:
    subgraphs = graph.copy()
    # First find all connecting atoms (assumes the center is node 0):
    connecting_atoms = list(subgraphs.neighbors(0))
    # Assert that we are removing 6 edges
    if len(connecting_atoms) != 6:
        raise ValueError(
            "First entry in the graph does not have 6 neighbors "
            "as expected for an octahedral complex."
        )
    # Then cut the graph by removing all connections to the first atom
    subgraphs.remove_edges_from([(0, c) for c in connecting_atoms])

    if equatorial_connecting_atoms is None:
        # Assume the first 4 connecting atoms belong to the equatorial ligands
        # and the other two are axial.
        equatorial_connecting_atoms = connecting_atoms[:4]
        axial_connecting_atoms = connecting_atoms[4:]
    else:
        axial_connecting_atoms = [
            c for c in connecting_atoms if c not in equatorial_connecting_atoms
        ]
        if len(equatorial_connecting_atoms) != 4 or len(axial_connecting_atoms) != 2:
            raise ValueError(
                "The provided equatorial connecting atoms "
                f"{equatorial_connecting_atoms} are not "
                "consistent with the neighbors of the first "
                f"entry in the graph {connecting_atoms}"
            )

    # Build lists of connecting atom and ligand
    # subgraph tuples by first finding set of nodes for the component that the
    # connecting atom c comes from (using nx.node_conncted_component()) and
    # then constructing a subgraph using this node set.
    axial_ligands = [
        (c, subgraphs.subgraph(nx.node_connected_component(subgraphs, c)))
        for c in axial_connecting_atoms
    ]
    equatorial_ligands = [
        (c, subgraphs.subgraph(nx.node_connected_component(subgraphs, c)))
        for c in equatorial_connecting_atoms
    ]

    # Note that the ligand centered RACs are averaged over the involved
    # ligands.
    # start = lc, scope = ax, product
    output[2] = np.mean(
        [
            atom_centered_AC(g, c, depth=depth, property_fun=property_fun)
            for (c, g) in axial_ligands
        ],
        axis=0,
    )

    # start = lc, scope = eq, product
    output[3] = np.mean(
        [
            atom_centered_AC(g, c, depth=depth, property_fun=property_fun)
            for (c, g) in equatorial_ligands
        ],
        axis=0,
    )

    output[4] = np.mean(
        [
            multi_centered_AC(g, depth=depth, property_fun=property_fun)
            for (_, g) in axial_ligands
        ],
        axis=0,
    )
    output[5] = np.mean(
        [
            multi_centered_AC(g, depth=depth, property_fun=property_fun)
            for (_, g) in equatorial_ligands
        ],
        axis=0,
    )

    # Finally calculate the difference ACs the same way:
    # start = mc, scope = all, difference
    output[6] = atom_centered_AC(
        graph, 0, depth=depth, operation=operator.sub, property_fun=property_fun
    )
    # start = lc, scope = ax, difference
    output[7] = np.mean(
        [
            atom_centered_AC(
                g, c, depth=depth, operation=operator.sub, property_fun=property_fun
            )
            for (c, g) in axial_ligands
        ],
        axis=0,
    )

    # start = lc, scope = eq, difference
    output[8] = np.mean(
        [
            atom_centered_AC(
                g, c, depth=depth, operation=operator.sub, property_fun=property_fun
            )
            for (c, g) in equatorial_ligands
        ],
        axis=0,
    )

    return output


def tetrahedral_racs(
    graph,
    depth: int = 3,
    property_fun=racs_property_vector,
):
    # For tetrahedrals there are 4 start/scope
    # combinations for product ACs and 2 for difference ACs.
    print(graph.nodes.keys())
    n_props = len(property_fun(graph, list(graph.nodes.keys())[0]))
    output = np.zeros((4 + 2, depth + 1, n_props))

    # start = f, scope = all, product
    output[0] = multi_centered_AC(graph, depth=depth, property_fun=property_fun)
    # start = mc, scope = all, product
    output[1] = atom_centered_AC(graph, 0, depth=depth, property_fun=property_fun)

    # For the other scopes the graph has to be subdivided into individual
    # ligand graphs. Make these changes on a copy of the graph:
    subgraphs = graph.copy()
    # First find all connecting atoms (assumes the center is node 0):
    connecting_atoms = list(subgraphs.neighbors(0))
    # Assert that we are removing 4 edges
    if len(connecting_atoms) != 4:
        raise ValueError(
            "First entry in the graph does not have 6 neighbors "
            "as expected for an octahedral complex."
        )
    # Then cut the graph by removing all connections to the first atom
    subgraphs.remove_edges_from([(0, c) for c in connecting_atoms])

    # Build lists of connecting atom and ligand
    # subgraph tuples by first finding set of nodes for the component that the
    # connecting atom c comes from (using nx.node_conncted_component()) and
    # then constructing a subgraph using this node set.
    ligands = [
        (c, subgraphs.subgraph(nx.node_connected_component(subgraphs, c)))
        for c in connecting_atoms
    ]

    # Note that the ligand centered RACs are averaged over the involved
    # ligands.
    # start = lc, scope = lig, product
    output[2] = np.mean(
        [
            atom_centered_AC(g, c, depth=depth, property_fun=property_fun)
            for (c, g) in ligands
        ],
        axis=0,
    )
    # start = lig, scope = lig, product
    output[3] = np.mean(
        [
            multi_centered_AC(g, depth=depth, property_fun=property_fun)
            for (_, g) in ligands
        ],
        axis=0,
    )

    # Finally calculate the difference ACs the same way:
    # start = mc, scope = all, difference
    output[4] = atom_centered_AC(
        graph, 0, depth=depth, operation=operator.sub, property_fun=property_fun
    )
    # start = lc, scope = ax, difference
    output[5] = np.mean(
        [
            atom_centered_AC(
                g, c, depth=depth, operation=operator.sub, property_fun=property_fun
            )
            for (c, g) in ligands
        ],
        axis=0,
    )
    return output
