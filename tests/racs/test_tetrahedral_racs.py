import numpy as np
from tmc_tools.graphs.constructors import graph_from_xyz_file
from tmc_tools.constants import atomic_numbers, electronegativity
from tmc_tools.graphs.racs import tetrahedral_racs


def test_tetrahedral_racs_vs_molSimplify(resource_path_root):

    rac_array = np.load(
        resource_path_root / "tetrahedral_racs" / "rac_features.npy", allow_pickle=True
    )

    # Since molSimplify uses different values for the covalent radii a
    # custom property function has to be used:
    """
    covalent_radii = {
        "H": 0.37,
        "C": 0.77,
        "N": 0.75,
        "O": 0.73,
        "S": 1.02,
        "F": 0.71,
        "Cr": 1.27,
        "Mn": 1.39,
        "Fe": 1.25,
        "Co": 1.26,
    }
    """
    covalent_radii = {
        1: 0.37,
        6: 0.77,
        7: 0.75,
        8: 0.73,
        9: 0.71,
        16: 1.02,
        24: 1.27,
        25: 1.39,
        26: 1.25,
        27: 1.26,
    }

    def property_fun(graph, node):
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

    for name, _, racs_ref in rac_array:
        graph = graph_from_xyz_file(
            resource_path_root / "tetrahedral_racs" / f"xyz_files/{name}.xyz"
        )
        racs = tetrahedral_racs(graph, depth=4, property_fun=property_fun)
        print(racs.flatten()[:20])
        print(racs_ref[5:25])
        np.testing.assert_allclose(sorted(racs.flatten()), sorted(racs_ref[5:]))
