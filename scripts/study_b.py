"""Studie (b): invloed van frequentie en geometrie op R, L en J_z."""

import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.solver import compute_rl,assemble_inductance_matrix, solve_rl
from src.geometry import make_mesh, UM
from src.physics import r_dc, skin_depth, freq_for_depth
from src.plotting import plot_current_density
from matplotlib.ticker import ScalarFormatter, NullFormatter
from matplotlib.colors import LogNorm

# Referentiegeometrie in micrometer
W, T, H = 4.0, 2.0, 3.0

# Gekozen meshwaarden 
NX, NY = 24, 16  

# Frequentiebereik voor sweep in Hz
F_MIN, F_MAX = 1e6, 1e11  

# Frequenties waarop J_maps worden geplot
FREQS_J = (1e6, 1e9, 1e10, 1e11)
FREQ_LABELS = {1e6: "1 MHz", 1e9: "1 GHz", 1e10: "10 GHz", 1e11: "100 GHz"}

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

def skin_check(freqs=(1e9, 1e10, 1e11), nx=24, ny=16):
    """
    Toon hoeveel cellen binnen één skindiepte van de rand liggen bij meshkeuze (nx=24,ny=16)
    Frequentieafhankelijk
    """
    mesh = make_mesh(W, T, H, nx=nx, ny=ny)
    for f in freqs:
        d = skin_depth(f)
        # xe/ye zijn de echte celranden (ongelijk verdeeld door clustering)
        # We tellen de randen binnen afstand d van de linker-/onderkant
        # n randen sluiten n-1 hele cellen in, vandaar de -1
        # Aan de andere kant is het aantal gelijk, want de cosinusclustering is symmetrisch
        n_x = int(np.sum(mesh.xe - mesh.xe[0] <= d)) - 1
        n_y = int(np.sum(mesh.ye - mesh.ye[0] <= d)) - 1
        # Als d >= w/2 of t/2 overlappen de lagen van beide kanten
        # en zegt het aantal niets meer (dan reikt het veld tot over het midden)
        print(f"f={f:.0e} Hz | delta = {d / UM:.3f} um | "
              f"kleinste cel x/y = {(mesh.xe[1] - mesh.xe[0]) / UM:.4f}/"
              f"{(mesh.ye[1] - mesh.ye[0]) / UM:.4f} um | "
              f"cellen in delta: x={n_x}, y={n_y}")

def freq_sweep(n_freq: int = 61):   
    """
    R(f), L(f), J(f) voor de referentiegeometrie
    Standaardwaarde n_freq: 61 frequentiewaarden in een sweep (12 intervallen per decade in standaardfrequentiebereik)
    """
    mesh = make_mesh(W, T, H, nx=NX, ny=NY)
    Lbar = assemble_inductance_matrix(mesh)   # één keer: hangt niet van f af
    freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), n_freq)
    return solve_rl(mesh, Lbar, freqs), mesh


def plot_freq_sweep(res, path="results/b_freq.png"):
    """R/R_DC (log-log) en L (semilogx) tegen f, met markeringen voor delta = t/2, w/2"""
    R_norm = res.R / r_dc(W, T)
    fig, (ax_R, ax_L) = plt.subplots(1, 2, figsize=(11, 4))

    ax_R.loglog(res.freqs, R_norm, label="R / R_DC")
    # hulplijn met helling 1/2 (skineffect: R ~ sqrt(f)), door het laatste punt
    f_guide = res.freqs[res.freqs >= 1e10]
    ax_R.loglog(f_guide, R_norm[-1] * np.sqrt(f_guide / res.freqs[-1]),
                color="tab:red", linewidth=1, label=r"helling 1/2 ($\propto\sqrt{f}$)")
    ax_R.set_ylabel("R / R_DC [-]")
    # leesbare y-ticks op de log-as
    ax_R.set_yticks([1, 1.5, 2, 3, 4])
    ax_R.yaxis.set_major_formatter(ScalarFormatter())
    ax_R.yaxis.set_minor_formatter(NullFormatter())

    ax_L.semilogx(res.freqs, res.L * 1e9)
    ax_L.set_ylabel("L [nH/m]")

    # Markeringen bij de frequenties waar delta = t/2 (y-richting) en delta = w/2 (x-richting).
    # Het veld komt van alle vier de zijden binnen, het midden is
    # pas afgeschermd als delta kleiner is dan beide halve afmetingen, dus bij
    # f > max(f(skindepth=t/2), f(skindepth=w/2)) begint stroom naar randen gedrukt te worden
    # Geleidelijke overgang, geen scherpe grens
    markers = [(r"$\delta = t/2$", T * UM / 2, "--"),
               (r"$\delta = w/2$", W * UM / 2, "-.")]
    for ax in (ax_R, ax_L):
        ax.set_xlabel("f [Hz]")
        ax.grid(True, which="both", alpha=0.3)
        for label, depth, style in markers:
            ax.axvline(freq_for_depth(depth), color="gray", linestyle=style,
                       label=label if ax is ax_R else None)
            
    ax_R.legend(loc="upper left", fontsize=8)

    Path(path).parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.show()



def current_density_maps(path="results/b_current.png"):
    """|J_z|/J_DC over de doorsnede bij enkele frequenties (log-schaal, grijs = uniform)"""
    mesh = make_mesh(W, T, H, nx=NX, ny=NY)
    Lbar = assemble_inductance_matrix(mesh)
    res = solve_rl(mesh, Lbar, FREQS_J)            # I_tot = 1 A
    J_dc = 1.0 / (W * UM * T * UM)                 # uniforme stroomdichtheid [A/m^2]
    J_norm = np.abs(res.J) / J_dc                  # rijen <-> frequenties
    lim = max(J_norm.max(), 1.0 / J_norm.min())    # symmetrisch rond 1 (op log-schaal)

    for f, Jn in zip(res.freqs, J_norm):
        print(f"f = {f:.0e} Hz | max |J|/J_DC = {Jn.max():6.2f} | min = {Jn.min():6.3f}")

    fig, axes = plt.subplots(1, len(FREQS_J), figsize=(14, 3.2), constrained_layout=True)
    for ax, f, J in zip(axes, res.freqs, res.J):
        pcm = plot_current_density(mesh, J / J_dc, ax, title=FREQ_LABELS.get(f, f"{f:.0e} Hz"))
        pcm.set_cmap("coolwarm")                            # blauw < 1 < rood, grijs = uniform
        pcm.set_norm(LogNorm(vmin=1.0 / lim, vmax=lim))     # 1 (grijs) in het midden van de log-schaal

    cbar = fig.colorbar(pcm, ax=axes.tolist(), label=r"$|J_z| / J_{DC}$")
    cbar.set_ticks([0.1, 0.3, 1, 3, 10])
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    cbar.ax.minorticks_off()

    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.show()

if __name__ == "__main__":
    # Een geschikte mesh vinden
    #mesh_check() 
    # Bij deze mesh controleren dat bij gekozen frequentiesbereik
    # de skindepth groter is dan een aantal cellen 
    #skin_check()  
    # Bij de gekozen mesh R(f) en I(f) bepalen over gekozen frequentiebereik
    res, mesh = freq_sweep() 
    # Frequetiesweep plotten
    #plot_freq_sweep(res)
    # Stroomdichtheid plotten bij enkele gekozen frequenties in frequentiebereik
    current_density_maps()