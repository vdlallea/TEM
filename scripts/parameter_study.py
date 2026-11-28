import numpy as np
import os
import matplotlib.pyplot as plt

from src.geometry import UM
from src.reference import l_external_reference
from src.solver import SIGMA_CU, compute_rl

basis = {"w": 4.0, "t": 2.0, "h": 3.0}   # µm
FREQS = np.logspace(7, 11, 17)          # 10 MHz .. 100 GHz
NX, NY = 30, 18


def varieer(naam, waarden):
    resultaten = {}
    for waarde in waarden:
        d = dict(basis)
        d[naam] = waarde
        res, mesh = compute_rl(w_um=d["w"], t_um=d["t"], h_um=d["h"],
                               freqs=FREQS, nx=NX, ny=NY, quad_order=4)

        R_dc = 1.0 / (SIGMA_CU * d["w"] * UM * d["t"] * UM)
        L_ref = l_external_reference(d["w"] * UM, d["h"] * UM, d["t"] * UM)

        print(f"\n{naam} = {waarde:g} µm   (w = {d['w']:g}, t = {d['t']:g}, h = {d['h']:g})")
        print(f"{'f [Hz]':>10}  {'R [Ohm/m]':>12}  {'L [nH/m]':>10}")
        for f, R, L in zip(res.freqs, res.R, res.L):
            print(f"{f:10.2e}  {R:12.4e}  {L * 1e9:10.4f}")

        resultaten[waarde] = (res, mesh, R_dc, L_ref)
    return resultaten


def plot_studie(naam, resultaten, ax_R, ax_L):
    for waarde, (res, mesh, R_dc, L_ref) in resultaten.items():
        label = f"{naam} = {waarde:g} µm"
        ax_R.semilogx(res.freqs, res.R / R_dc, "o-", ms=3, label=label)
        lijn, = ax_L.semilogx(res.freqs, res.L * 1e9, "o-", ms=3, label=label)
        # Hammerstad-referentie als stippellijn in dezelfde kleur
        ax_L.axhline(L_ref * 1e9, ls="--", lw=1, color=lijn.get_color())

    vast = ", ".join(f"{k} = {v:g}" for k, v in basis.items() if k != naam)
    ax_R.set_title(f"variërende {naam}  ({vast} µm)")
    ax_R.set_ylabel(r"$R / R_{DC}$")
    ax_L.set_ylabel("L [nH/m]")
    ax_L.set_xlabel("f [Hz]")
    for ax in (ax_R, ax_L):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)


def main():
    waarden = [1.0, 2.0, 5.0, 10.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for kolom, naam in enumerate(["w", "t", "h"]):
        resultaten = varieer(naam, waarden)
        plot_studie(naam, resultaten, axes[0, kolom], axes[1, kolom])
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/parameter_study.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()


    


