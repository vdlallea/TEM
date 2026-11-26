#Hulpfuncties voor studie deel 4b
#Berekenen skindepth en r_dc op basis van gekozen parameters
import numpy as np
from .solver import SIGMA_CU,MU0

def skin_depth(f, sigma : float= SIGMA_CU):
    """Berekening skindepth in meter
    Frequentie in Hz, mag scalair of numpy-array zijn
    Hoe diep dringt H-veld bij deze frequentie door in koper
    Vergelijk met t en w om te beoordelen of effect invloed heeft
    Curus formule 5.27
    """
    return 1.0/np.sqrt(np.pi* np.asarray(f, dtype=float) * MU0 * sigma)

def r_dc(w_um: float, t_um: float, sigma: float = SIGMA_CU) -> float:
    """Gelijkstroomweerstand per meter
    Afmetingen in micrometer, zoals in compute_rl.
    R(f) benadert bij lage frequentie deze waarde (uniforme stroomdichtheid).
    """
    return 1.0 / (sigma * (w_um * 1e-6) * (t_um * 1e-6))

def freq_for_depth(depth_m, sigma: float = SIGMA_CU):
    """Frequentie waarbij de skindiepte gelijk is aan depth_m (meter)"""
    return 1.0 / (np.pi * MU0 * sigma * np.asarray(depth_m, dtype=float) ** 2)