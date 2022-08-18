import pytest
import pickle
import numpy as np
from tmc_tools.graphs.constructors import graph_from_xyz_file
from tmc_tools.constants import atomic_numbers, electronegativity
from tmc_tools.graphs.racs import ocatahedral_racs


@pytest.mark.parametrize(
    "xyz_path, ref_path, eq_atoms",
    [
        ("fe_carbonyl_6.xyz", "racs_Fe_carbonyl_6.pickle", None),
        (
            "mn_furan_water_ammonia_furan_water_ammonia.xyz",
            "racs_Mn_furan_water_ammonia_furan_water_ammonia.pickle",
            None,
        ),
        # Commented out because molSimplify handles triple bidentates weirdly
        # ('cr_acac_acac_bipy.xyz',
        #  'racs_Cr_acac_acac_bipy.pickle', [6, 20, 29, 30]),
        (
            "co_acac_en_water_hydrogensulfide.xyz",
            "racs_Co_acac_en_water_hydrogensulfide.pickle",
            None,
        ),
    ],
)
def test_octahedral_racs_vs_molSimplify(
    resource_path_root, xyz_path, ref_path, eq_atoms, atol=1e-4
):

    graph = graph_from_xyz_file(resource_path_root / xyz_path)

    with open(resource_path_root / ref_path, "rb") as fin:
        ref_dict = pickle.load(fin)

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

    depth = 3
    properties = ["Z", "chi", "T", "I", "S"]
    descriptors = ocatahedral_racs(
        graph,
        depth=depth,
        equatorial_connecting_atoms=eq_atoms,
        property_fun=property_fun,
    )

    # Dictionary encoded the order of the descriptors in the numpy array
    start_scopes = {
        0: ("f", "all"),
        1: ("mc", "all"),
        2: ("lc", "ax"),
        3: ("lc", "eq"),
        4: ("f", "ax"),
        5: ("f", "eq"),
        6: ("D_mc", "all"),
        7: ("D_lc", "ax"),
        8: ("D_lc", "eq"),
    }

    for s, (start, scope) in start_scopes.items():
        for d in range(depth + 1):
            for p, prop in enumerate(properties):
                print(
                    start,
                    scope,
                    d,
                    prop,
                    descriptors[s, d, p],
                    ref_dict[f"{start}-{prop}-{d}-{scope}"],
                )
                assert (
                    abs(descriptors[s, d, p] - ref_dict[f"{start}-{prop}-{d}-{scope}"])
                    < atol
                )
