#convergentie bestuderen van R en L bij vaste frequentie
#Twee situaties bekijken:
#1) rooster verfijnen (nx, ny), quad order vast
#2) verfijning van kwadratuur (quad_order), rooster vast.
#referentie is telkens fijnste manier van instellen: meet de convergentie, niet absolute nauwkeurigheid

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from src.solver import compute_rl

W_UM, T_UM, H_UM = 4.0, 2.0 , 3.0
FREQ = 10e9 

MESH_SIZES = [(4, 3), (6, 4), (8, 5), (10, 6), (14, 8), (18, 10), (24, 14), (30, 18), (36, 22)]
QUAD_FIXED = 4
QUAD_REF = 12
MESH_FIXED = (10,6)

#return (R, L, rekentijd) voor een bepaalde discretisatie
def run(nx, ny, quad_order):
    t0 = time.perf_counter() #hoogst mogelijke tijdsreolutie voor korte tijdsduren
    res, _ = compute_rl(W_UM, T_UM, H_UM, [FREQ], nx=nx, ny=ny, quad_order=quad_order)
    return res.R[0], res.L[0], time.perf_counter()-t0

#Berekent R en L voor elk rooster, plus het verschil met het vorige rooster
def mesh_study():
    raw = []
    for nx, ny in MESH_SIZES:
        R, L, dt = run(nx, ny, QUAD_FIXED)
        raw.append((nx, ny, R, L, dt))

    #referentie: het fijnste rooster
    R_ref, L_ref = raw[-1][2], raw[-1][3]

    rows = []
    for i, (nx, ny, R, L, dt) in enumerate(raw):
        eR = abs(R - R_ref) / R_ref          #fout tegenover referentie
        eL = abs(L - L_ref) / L_ref
        if i == 0:
            dR = dL = np.nan                 #geen vorig rooster
        else:
            dR = abs(R - raw[i - 1][2]) / R  #verschil met vorige verfijning
            dL = abs(L - raw[i - 1][3]) / L
        rows.append((nx, ny, nx * ny, R, L, eR, eL, dR, dL, dt))
    return rows
    
def quad_study():
    R_ref, L_ref, _ = run(*MESH_FIXED, QUAD_REF)
    rows = []
    for q in [2, 3, 4, 6, 8]:
        R, L, dt = run(*MESH_FIXED, q)
        rows.append((q, R, L, abs(R-R_ref)/R_ref, abs(L-L_ref)/L_ref, dt))
    return rows

def main():
    print("Situatie 1: roosterverfijning (quad_order = %d)" %QUAD_FIXED)
    print(f"{'nx':>4} {'ny':>4} {'N':>5} {'R [Ohm/m]':>12} {'L [nH/m]':>10} {'fout R':>10} {'fout L':>10} {'dR':>10} {'dL':>10} {'tijd [s]':>9}")
    mrows = mesh_study()
    for nx, ny, N, R, L, eR, eL, dR, dL, dt in mrows:
        print(f"{nx:4d} {ny:4d} {N:5d} {R:12.4e} {L*1e9:10.4f} {eR:10.2e} {eL:10.2e} {dR:10.2e} {dL:10.2e} {dt:9.3f}")

    print(f"\nSituatie 2: kwadratuurverfijning (rooster = {MESH_FIXED})")
    print(f"{'q':>4} {'R [Ohm/m]':>12} {'L [nH/m]':>10} {'fout R':>10} {'fout L':>10} {'tijd [s]':>9}")
    qrows = quad_study()
    for q, R, L, eR, eL, dt in qrows:
        print(f"{q:4d} {R:12.4e} {L*1e9:10.4f} {eR:10.2e} {eL:10.2e} {dt:9.3f}")

    #plotten: verschil met vorig rooster tegenover het aantal cellen (log-log), eerste rooster weglaten
    N = np.array([r[2] for r in mrows[1:]])
    dR = np.array([r[7] for r in mrows[1:]])
    dL = np.array([r[8] for r in mrows[1:]])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(N, dR, "o-", label="R")
    ax.loglog(N, dL, "s-", label="L")
    ax.axhline(1e-3, ls="--", color="gray", label="drempel 0,1 %")
    ax.set_xlabel("aantal cellen N")
    ax.set_ylabel("relatief verschil met vorig rooster")
    ax.set_title("Convergentie bij 10 GHz")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/convergence.png", dpi=150)
    plt.show()
    
if __name__ == "__main__":
    main()