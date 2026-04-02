"""
geometry/sections.py
Définitions des formes de sections transversales.
"""
import numpy as np
from abc import ABC, abstractmethod

class BaseSection(ABC):
    @abstractmethod
    def generate_points(self, num_points: int) -> np.ndarray:
        """Génère les points du contour en coordonnées locales (u, v)."""
        pass
    
    @abstractmethod
    def contains_point(self, u: float, v: float) -> bool:
        """Vérifie si un point local (u, v) est à l'intérieur de la section."""
        pass

    @abstractmethod
    def get_area(self) -> float:
        pass

class SuperEllipticalSection(BaseSection):
    def __init__(self, a: float, b: float, n: float = 2.5):
        self.a = a # Demi-axe majeur
        self.b = b # Demi-axe mineur
        self.n = n # Exposant (2=ellipse, >2=rectangulaire)
        
    def generate_points(self, num_points: int) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
        
        # Formule paramétrique signée pour préserver les quadrants
        c = np.cos(theta)
        s = np.sin(theta)
        
        x = self.a * np.sign(c) * (np.abs(c) ** (2 / self.n))
        y = self.b * np.sign(s) * (np.abs(s) ** (2 / self.n))
        
        return np.column_stack([x, y, np.zeros(num_points)])

    def contains_point(self, u: float, v: float) -> bool:
        # Équation implicite : |x/a|^n + |y/b|^n <= 1
        return (abs(u)/self.a)**self.n + (abs(v)/self.b)**self.n <= 1.0

    def get_area(self) -> float:
        from scipy.special import gamma
        return 4 * self.a * self.b * (gamma(1 + 1/self.n)**2) / gamma(1 + 2/self.n)

class CircularSection(SuperEllipticalSection):
    def __init__(self, radius: float):
        super().__init__(radius, radius, 2.0)

def create_section(type_str: str, **kwargs):
    # On récupère une valeur de base pour le rayon au cas où
    r_default = kwargs.get('radius', 1.0)
    
    if type_str == 'circular':
        return CircularSection(r_default)
        
    elif type_str == 'superelliptical':
        # On extrait les paramètres proprement
        a = kwargs.get('major_radius', r_default)
        # Si minor_radius n'est pas fourni, on prend 70% de a (valeur arbitraire cohérente)
        b = kwargs.get('minor_radius', a * 0.7) 
        n = kwargs.get('exponent', 2.5)
        return SuperEllipticalSection(a, b, n)
        
    raise ValueError(f"Type de section inconnu: {type_str}")