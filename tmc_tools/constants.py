import numpy as np
from ase.data import chemical_symbols, atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors


metal_list = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
metal_list = np.array(list(range(21, 30))+list(range(39, 48))+list(range(72, 80)))

__all__ = ["chemical_symbols", "atomic_numbers", "covalent_radii", "jmol_colors"]

roman_numerals = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}

# From:
# https://en.wikipedia.org/wiki/Electronegativities_of_the_elements_(data_page)
no_data = np.NaN
electronegativity = np.array(
    [
        no_data,  # X
        2.20,  # H
        no_data,  # He
        0.98,  # Li
        1.57,  # Be
        2.04,  # B
        2.55,  # C
        3.04,  # N
        3.44,  # O
        3.98,  # F
        no_data,  # Ne
        0.93,  # Na
        1.31,  # Mg
        1.61,  # Al
        1.90,  # Si
        2.19,  # P
        2.58,  # S
        3.16,  # Cl
        no_data,  # Ar
        0.82,  # K
        1.00,  # Ca
        1.36,  # Sc
        1.54,  # Ti
        1.63,  # V
        1.66,  # Cr
        1.55,  # Mn
        1.83,  # Fe
        1.88,  # Co
        1.91,  # Ni
        1.90,  # Cu
        1.65,  # Zn
        1.81,  # Ga
        2.01,  # Ge
        2.18,  # As
        2.55,  # Se
        2.96,  # Br
        3.00,  # Kr
        0.82,  # Rb
        0.95,  # Sr
        1.22,  # Y
        1.33,  # Zr
        1.60,  # Nb
        2.16,  # Mo
        1.90,  # Tc
        2.20,  # Ru
        2.28,  # Rh
        2.20,  # Pd
        1.93,  # Ag
        1.69,  # Cd
        1.78,  # In
        1.96,  # Sn
        2.05,  # Sb
        2.10,  # Te
        2.66,  # I
        2.60,  # Xe
        0.79,  # Cs
        0.89,  # Ba
        1.10,  # La
        1.12,  # Ce
        1.13,  # Pr
        1.14,  # Nd
        no_data,  # Pm
        1.17,  # Sm
        no_data,  # Eu
        1.20,  # Gd
        no_data,  # Tb
        1.22,  # Dy
        1.23,  # Ho
        1.24,  # Er
        1.25,  # Tm
        no_data,  # Yb
        1.27,  # Lu
        1.30,  # Hf
        1.50,  # Ta
        2.36,  # W
        1.90,  # Re
        2.20,  # Os
        2.20,  # Ir
        2.28,  # Pt
        2.54,  # Au
        2.00,  # Hg
        1.62,  # Tl
        2.33,  # Pb
        2.02,  # Bi
        2.00,  # Po
        2.20,  # At
        no_data,  # Rn
        no_data,  # Fr
        0.90,  # Ra
        1.10,  # Ac
        1.30,  # Th
        1.50,  # Pa
        1.38,  # U
        1.36,  # Np
        1.28,  # Pu
        1.30,  # Am
        1.30,  # Cm
        1.30,  # Bk
        1.30,  # Cf
        1.30,  # Es
        1.30,  # Fm
        1.30,  # Md
        1.30,  # No
    ]
)
