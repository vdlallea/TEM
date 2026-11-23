#Hulpfuncties voor studie deel 4b
#Berekenen skindepth en r_dc op basis van gekozen parameters
import numpy as np
from .solver import SIGMA_CU,MU0

def skin_depth(f, sigma : float= SIGMA_CU):
    """Berekening skindepth in meter
    Frequentie in Hz, mag scalair of numpy-array zijn
    Hoe diep dringt stroom bij deze frequentie door in koper
    Vergelijk met t en w om te beoordelen of effect invloed heeft
    Curus formule 5.27
    """
    return 1.0/np.sqrt(np.pi* np.asarray(f, dtype=float) * MU0 * sigma)

def r_dc(w_um: float, t_um: float, sigma: float = SIGMA_CU) -> float:
    """Gelijkstroomweerstand per meter
    Afmetingen in micrometer, zoals in compute_rl. Dit is de waarde waar
    R(f) bij lage frequentie naartoe gaat (uniforme stroomdichtheid).
    """
    return 1.0 / (sigma * (w_um * 1e-6) * (t_um * 1e-6))