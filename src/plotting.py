#plotten van R(f), L(f) en de stroomdichtheid in de geleider

import numpy as np
from .geometry import UM

def plot_RL(res, ax_R, ax_L):
    ax_R.semilogx(res.freqs, res.R, "o-")
    ax_R.set_xlabel("f [Hz]")
    ax_R.set_ylabel(r"R [$\Omega$/m]")
    ax_R.grid(True, which="both", alpha=0.3)

    ax_L.semilogx(res.freqs, res.L * 1e9, "o-")
    ax_L.set_xlabel("f [Hz]")
    ax_L.set_ylabel("L [nH/m]")
    ax_L.grid(True, which="both", alpha=0.3)
   
#plot |J_z| over de doorsnede van de geleider (een waarde per cel)
def plot_current_density(mesh, J, ax, title: str = ""):
    Jgrid = np.abs(J).reshape(mesh.nx, mesh.ny)
    pcm = ax.pcolormesh(mesh.xe / UM, mesh.ye / UM, Jgrid.T, shading="flat")
    ax.set_aspect("equal")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(title)
    return pcm     