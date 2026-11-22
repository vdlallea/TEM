#Logaritmische-kern integralen over rechthoeken.

#Berekent de integraal van ln|(x, y) - (x', y')| over x' in [x3, x4] en y' in [y3, y4].

#Aanpak: de x'-integraal wordt analytisch uitgevoerd, de y'-integraal numeriek
#met Gauss-Legendre. Een volledig analytische 2D-formule met arctan2 is niet
#bruikbaar, omdat de primitieve een takkenlijn heeft waarlangs de waarde springt (Appendix B).

import numpy as np

#Primitieve van ln sqrt(s^2 + a^2) naar s, voor a > 0 (continu in s)
def _F(s, a):
    r2 = s * s + a * a
    return s * 0.5 * np.log(r2) - s + a * np.arctan(s / a)

#Integraal van ln|(x, y) - (x', y')| over de rechthoek [x3, x4] x [y3, y4].
#x en y mogen numpy-arrays van willekeurige (broadcastbare) vorm zijn; het
#resultaat heeft dezelfde vorm. Eenheden zijn willekeurig maar moeten
#consistent zijn. Er wordt aangenomen dat x3 < x4 en y3 < y4.
def rect_log_potential(x, y, x3, x4, y3, y4, n_gauss: int = 10):
    if n_gauss < 1:
        raise ValueError("n_gauss moet minstens 1 zijn")

    t, w = np.polynomial.legendre.leggauss(n_gauss)
    yp = 0.5 * (y3 + y4) + 0.5 * (y4 - y3) * t   # knooppunten voor y'
    wy = 0.5 * (y4 - y3) * w                     # bijbehorende gewichten

    # kleine ondergrens voorkomt a = 0 wanneer het observatiepunt op een
    # bronlijn ligt (log-singulariteit, integreerbaar)
    eps = 1e-12 * ((x4 - x3) + (y4 - y3))

    x = np.asarray(x, dtype=float)[..., None]
    y = np.asarray(y, dtype=float)[..., None]
    a = np.maximum(np.abs(y - yp), eps)

    # x'-integraal: F(x - x3, a) - F(x - x4, a)
    inner = _F(x - x3, a) - _F(x - x4, a)
    return inner @ wy