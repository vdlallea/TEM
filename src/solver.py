#Volume integraal oplosser voor p.u.l weerstand R en inductantie L
#Uit Galerkin-weging: Rbar*I + jomega*Lbar*I = (R+jomega*L) * I_tot*alpha
#met Rbar diagonaalmatrix obv oppervlakte alpha en geleidbaarheid sigma -> weerstand per cel
#met Lbar inductiematrix: magnetische wisselwerking tussen alle cellen
#Samen met I_tot = som over alle n van I_n * alpha_n krijgt men:
#    Z = R + j*omega*L = 1 / (alpha^T M^-1 alpha),   M = Rbar + j*omega*Lbar,
#    I = Z * I_tot * M^-1 alpha.


from dataclasses import dataclass
import numpy as np
from .geometry import make_mesh
from .kernels import rect_log_potential

#Mogelijk uitbreiding nog toe te voegen, koper geldt MUr ongeveer 1, dus MU=MU0
#Andere materialen eenvoudig aanpassen door te vermenigvuldigen met MUr
MU0 = 4e-7 *np.pi 
SIGMA_CU = 5.8e7


@dataclass
class Result:
    freqs: np.ndarray #Hz, 1D
    R: np.ndarray #Ohm/m, 1D freq afh
    L: np.ndarray #H/m,  1D freq afh
    J: np.ndarray # A/m^2, complex, rijen<->freq, klommen<->N cellen


#Lbar (NxN, freq onafh)
#Observatie-integraal (over S_m): numeriek via Gauss-Legendre kwadratuur (orde quad_order)
#Bron-integraal (over S_n): exact berekend in gesloten vorm (analytisch)
def assemble_inductance_matrix(mesh, quad_order: int = 4) -> np.ndarray:
    if quad_order<1:
        raise ValueError("quad_order must be at least 1")
    
    gp, gw = np.polynomial.legendre.leggauss(quad_order)
    gi, gj = np.meshgrid(gp, gp, indexing="ij")
    wi, wj = np.meshgrid(gw, gw, indexing="ij")
    qs = gi.ravel() #referentie coördinaten [-1, 1]
    qt = gj.ravel()
    qw = (wi * wj).ravel() / 4.0 #gewichten sommeren tot 1 (genormaliseerd tot de cel opp)    
    
    #observatiepunten voor elke testcel
    xc = 0.5 * (mesh.x1 + mesh.x2)
    hx = 0.5 * (mesh.x2 - mesh.x1)
    yc = 0.5 * (mesh.y1 + mesh.y2)
    hy = 0.5 * (mesh.y2 - mesh.y1)
    X = xc[:, None] + hx[:, None] * qs[None, :]
    Y = yc[:, None] + hy[:, None] * qt[None, :]

    alpha = mesh.alpha
    N = mesh.n_cells
    K = np.empty((N, N))
    
    for n in range(N):
        x3, x4 = mesh.x1[n], mesh.x2[n]
        y3, y4 = mesh.y1[n], mesh.y2[n]
        #directe bron: integraal over S_n van ln|r - r'|
        direct = rect_log_potential(X, Y, x3, x4, y3, y4)
        #spiegelbron: integraal over S_n van ln|r - r''|, met r'' = (x', -y')
        mirror = rect_log_potential(X, Y, x3, x4, -y4, -y3)
        # Observatie-integraal over S_m via Gauss-kwadratuur: alpha_m * sum_q w_q f(r_mq)
        K[:, n] = alpha * ((direct - mirror) @ qw)

    return -MU0 / (2.0 * np.pi) * K


#Los op voor R(f), L(f) en de stroomdichtheid J(f) op elke frequentie
def solve_rl(mesh, Lbar: np.ndarray, freqs, sigma: float = SIGMA_CU, I_tot: float = 1.0) -> Result:
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if np.any(freqs <= 0.0):
        raise ValueError("Alle frequenties moeten >0 Hz (gebruik een kleine waarde voor DC)")
    
    alpha = mesh.alpha
    R_list, L_list, J_list = [], [], []
    
    for f in freqs:
        omega = 2.0 * np.pi * f
        M = np.diag(alpha / sigma) + 1j * omega * Lbar
        x = np.linalg.solve(M, alpha.astype(complex))   # M^-1 alpha
        Z = 1.0 / np.sum(alpha * x)                     # R + j omega L (Ohm/m)
        J_list.append(Z * I_tot * x)                    #stroomdichtheid per cel
        R_list.append(Z.real)
        L_list.append(Z.imag / omega)  
        
    return Result(freqs=freqs, R=np.array(R_list), L=np.array(L_list), J=np.array(J_list))


def compute_rl(w_um: float, t_um: float, h_um: float, freqs, nx: int = 10, ny: int = 6, quad_order: int = 4, clustered: bool = True):
    #Hulpfunctie (wrapper): invoergeometrie in micrometer, returns (Result, Mesh)
    mesh = make_mesh(w_um, t_um, h_um, nx=nx, ny=ny, clustered=clustered)
    Lbar = assemble_inductance_matrix(mesh, quad_order=quad_order)
    res = solve_rl(mesh, Lbar, freqs)
    return res, mesh   

