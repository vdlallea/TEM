import numpy as np
from src.physics import skin_depth, r_dc, freq_for_depth
from src.geometry import UM


def test_skin_depth_copper_1ghz():
    # gekend: ongeveer 2.09 micrometer voor koper bij 1 GHz
    assert np.isclose(skin_depth(1e9) / UM, 2.09, rtol=1e-2)


def test_r_dc_value():
    # w=4 um, t=2 um -> 1/(5.8e7 * 8e-12) = 2155.17 Ohm/m
    assert np.isclose(r_dc(4.0, 2.0), 2155.17, rtol=1e-4)

def test_freq_for_depth_roundtrip():
    f = 3.7e9
    assert np.isclose(freq_for_depth(skin_depth(f)), f, rtol=1e-12)