#voorbeeld/test: : R(f), L(f) en stroomdensiteit voor 1 microstripgeometrie
import os

import matplotlib.pyplot as plt
import numpy as np

from src.geometry import UM
from src.plotting import plot_current_density, plot_RL
from src.reference import l_external_reference
from src.solver import SIGMA_CU, compute_rl


def main():
    w_um, t_um, h_um = 4.0, 2.0, 3.0
    freqs = np.logspace(7, 11, 9)   #10MHz tot 100GHz

    res, mesh = compute_rl(w_um, t_um, h_um, freqs, nx=10, ny=6, quad_order=4)

    print(f"{'f [Hz]':>10}  {'R [Ohm/m]':>12}  {'L [nH/m]':>10}")
    for f, R, L in zip(res.freqs, res.R, res.L):
        print(f"{f:10.2e}  {R:12.4e}  {L * 1e9:10.4f}")

    R_dc = 1.0 / (SIGMA_CU * w_um * UM * t_um * UM)
    L_ref = l_external_reference(w_um * UM, h_um * UM, t_um * UM)
    print(f"\nDC weerstand (analytisch):           {R_dc:.4e} Ohm/m")
    print(f"Externe L, Hammerstad (benadering.):   {L_ref * 1e9:.4f} nH/m")
    print("Bij hoge frequentie moet L naar de externe waarde gaan (Hammerstad).")
    print("Bij lage frequentie is L groter, omdat de interne inductie erbij komt.")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_RL(res, axes[0], axes[1])
    pcm1 = plot_current_density(mesh, res.J[0], axes[2],
                                title=f"|J| op {res.freqs[0]:.1e} Hz")
    pcm2 = plot_current_density(mesh, res.J[-1], axes[3],
                                title=f"|J| op {res.freqs[-1]:.1e} Hz")
    fig.colorbar(pcm1, ax=axes[2])
    fig.colorbar(pcm2, ax=axes[3])
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/example.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()