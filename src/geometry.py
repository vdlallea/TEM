#discretisatie van koperdoorsnede in rechthoekige cellen.
#cellen naar de randen toe kleiner maken, aangezien de stroom zich daar concentreert


from dataclasses import dataclass
#gebruik dit om eenvoudige klassen te maken die voornamelijk data opslaan
import numpy as np

UM = 1e-6 #meter per micrometer

@dataclass(frozen=True) #klasse onveranderlijk maken na aanmaak
class Mesh:
    #rechthoekige cellen die de doorsnede bedekken (meters gebruiken)
    #index van een cel: k=i*ny + j (i<->x, j<->y)

    x1: np.ndarray #linkerkant can elke cel
    x2: np.ndarray #rechterkant
    y1: np.ndarray #onderkant
    y2: np.ndarray #bovenkant
    xe: np.ndarray #1D x-randen, lengte nx+1 (om te plotten)
    ye: np.ndarray #1D y-eanden, lengte ny+1
    nx: int
    ny: int
    
    @property
    def n_cells(self) -> int:
        return self.x1.size
    
    @property
    def alpha(self) ->np.ndarray: #type hint
        #celoppervlakken alpha_n in vierkante matrix
        return (self.x2 - self.x1) * (self.y2 - self.y1)


#_ voor functie duidt op intern gebruik van de functie 
#coordinaten berekenen tussen een start en stopwaarde, verdeeld in n intervallen
#als clustered true is-> punten dicht bij de randen adhv een cosinustransformatie
def _edges(start: float, stop: float, n: int, clustered: bool) ->np.ndarray:
    s = np.linspace(0.0, 1.0, n+1)
    if clustered:
        s = 0.5 * (1.0-np.cos(np.pi*s))
    return start + (stop-start)*s

#meshen van conductor van microstrip
#geleider:  x in [-w/2, w/2], y in [h, h+t]. PEC grondvlak op y=0, dimensies in micrometer
def make_mesh(w_um: float, t_um: float, h_um: float, nx: int = 10, ny: int = 6, clustered: bool =True) -> Mesh:
    if nx<1 or ny<1:
        raise ValueError("nx and ny must be at least 1")
    w, t, h = w_um*UM, t_um*UM, h_um*UM
    xe = _edges(-0.5*w, 0.5*w, nx, clustered)
    ye = _edges(h, h+t, ny, clustered)
    
    X1, Y1 = np.meshgrid(xe[:-1], ye[:-1], indexing="ij") #2D-rooster maken obv losse 1D-lijsten
    X2, Y2 = np.meshgrid(xe[1:], ye[1:], indexing="ij")
    #.ravel: meerdimensionaal->1D array
    return Mesh(x1=X1.ravel(), x2=X2.ravel(), y1=Y1.ravel(), y2=Y2.ravel(), xe=xe, ye=ye, nx=nx, ny=ny)