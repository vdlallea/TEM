import numpy as np
#om numeriek een dubbele gedefinieerde integraal te berekenen over een 2D-ruimte
from scipy.integrate import dblquad
import pytest
from src.geometry import make_mesh
from src.kernels import rect_log_potential
from src.solver import SIGMA_CU, assemble_inductance_matrix, solve_rl


#observatiepunten rond de bronrechthoek [-1, 1] x [2, 3], zowel links, rechts, boven als onder
@pytest.mark.parametrize("x, y", [
    (3.0, 0.5),    # rechts, onder de bron
    (-3.0, 0.5),   # links, onder de bron
    (0.0, 0.5),    # recht onder de bron
    (1.5, 4.0),    # boven de bron, rechts van het midden
    (-0.5, 4.0),   # boven de bron, links van het midden
    (0.0, 1.0),    # dicht onder de bron
])
def test_rect_potential_matches_quadrature(x, y):
    x3, x4, y3, y4 = -1.0, 1.0, 2.0, 3.0
    ref, _ = dblquad(lambda yp, xp: np.log(np.hypot(x - xp, y - yp)),
                     x3, x4, y3, y4)
    val = rect_log_potential(np.array(x), np.array(y), x3, x4, y3, y4)
    assert np.isclose(val, ref, rtol=1e-5, atol=1e-8)


#op heel lage frequenties is de stroom unifrom en R=1/(sigma*w*t)
def test_dc_resistance():
    mesh = make_mesh(4.0, 2.0, 3.0, nx=8, ny=4)
    Lbar = assemble_inductance_matrix(mesh)
    res = solve_rl(mesh, Lbar, [1.0])
    R_dc = 1.0 / (SIGMA_CU * 4e-6 * 2e-6)
    assert np.isclose(res.R[0], R_dc, rtol=1e-6)


def test_lbar_is_positive_and_nearly_symmetric():
    mesh = make_mesh(4.0, 2.0, 3.0, nx=6, ny=4)
    Lbar = assemble_inductance_matrix(mesh)
    #de diagonaal moet positief zijn
    assert np.all(np.diag(Lbar) > 0)
    #symmetrie geldt analytisch, kwadratuur breekt deze slechts in geringe mate
    assert np.allclose(Lbar, Lbar.T, rtol=1e-2, atol=0.0)