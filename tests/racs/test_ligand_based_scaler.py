#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright 2021, Jonas Oldenstaedt <joldenstaedt@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.
#
# =============================================================================

# =============================================================================
# Imports
# =============================================================================

import numpy as np
from tmc_tools.graphs.racs import get_set_of_lig_scaled_RACs

def test_ligand_based_scaling_function(resource_path_root):
    dummy = [[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [0.5, 0.5, 0.5]], [[1.5,1.5,1.5], [1.5,1.5, 1.5], [0.5,0.5,0.5]]]
    dummy_lig = [[1,1,0], [0,2,2]]
    rac = get_set_of_lig_scaled_RACs(dummy, dummy_lig)
    np.testing.assert_allclose(rac, np.array([[[-1., -1., -1.], [ 1.,  1.,  1.], [-1., -1., -1.]], [[ 1.,  1.,  1.], [ 1.,  1., 1.], [-1., -1., -1.]]]).reshape(-1, 9))
