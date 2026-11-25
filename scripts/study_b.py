"""Studie (b): invloed van frequentie en geometrie op R, L en J_z."""

import time
import numpy as np
from src.solver import compute_rl

# Referentiegeometrie in micrometer
W, T, H = 4.0, 2.0, 3.0


def mesh_check(freqs=(1e9, 1e11), meshes=((8, 6), (16, 12), (24, 16), (32, 22), (40, 28))):
    """
    Toon R en L voor steeds fijnere meshes (nx, ny) bij frequentie= 1 GHz en = 100 GHz
    Geeft indicatie over extra nauwkeurigheid in trade-off met rekentijd
    """

    for nx, ny in meshes:
        t0 = time.time()
        res, _ = compute_rl(W, T, H, list(freqs), nx=nx, ny=ny)
        dt = time.time() - t0
        print(f"nx={nx:2d} ny={ny:2d} N={nx*ny:4d} | "
              f"R = {np.round(res.R, 1)} Ohm/m | "
              f"L = {np.round(res.L * 1e9, 2)} nH/m | {dt:.2f} s")

import numpy as np
from src.geometry import make_mesh, UM
from src.physics import skin_depth


def skin_check(freqs=(1e9, 1e10, 1e11), nx=24, ny=16):
    """
    Toon hoeveel cellen binnen één skindiepte van de rand liggen bij meshkeuze (nx=24,ny=16)
    """
    mesh = make_mesh(W, T, H, nx=nx, ny=ny)
    for f in freqs:
        d = skin_depth(f)
        n_x = int(np.sum(mesh.xe - mesh.xe[0] <= d)) - 1
        n_y = int(np.sum(mesh.ye - mesh.ye[0] <= d)) - 1
        print(f"f={f:.0e} Hz | delta = {d / UM:.3f} um | "
              f"kleinste cel x/y = {(mesh.xe[1] - mesh.xe[0]) / UM:.4f}/"
              f"{(mesh.ye[1] - mesh.ye[0]) / UM:.4f} um | "
              f"cellen in delta: x={n_x}, y={n_y}")
    
if __name__ == "__main__":
    mesh_check()
    skin_check()