#benaderde analytische schatting van de externe inductie
#Hammerstad's microstripformule voor lijnen gevuld met lucht (eps_r=1)
#ook diktecorrectie van Wheeler gebruiken (benadering, niet exact)
#inductieporbleem onafhankelijk van de permittiviteit van het substraat 
#->L_ext = Z0_lucht/c is een bruikbare referentie voor de externe inductie.

import numpy as np

C0 = 299792458.0  # m/s

#karakteristieke impedantie van microstrip in lucht, input in meter
def z0_air_hammerstad(w: float, h: float, t: float) -> float:
    we = w + (t / np.pi) * (1.0 + np.log(2.0 * h / t))   # effective breedte
    u = we / h
    if u <= 1.0:
        return 60.0 * np.log(8.0 / u + u / 4.0)
    return 120.0 * np.pi / (u + 1.393 + 0.667 * np.log(u + 1.444))


#externe p.u.l inductantie: L=Z0/c voor een luchtlijn
def l_external_reference(w: float, h: float, t: float) -> float:
    return z0_air_hammerstad(w, h, t) / C0