"""!
@file sections.py
@brief Definitions of cross-sectional shapes for fibers.
@details Provides a hierarchy of classes to define 2D profiles (Circular, Super-elliptical) 
that can be extruded along a 3D curve.
"""
import numpy as np
from abc import ABC, abstractmethod

class BaseSection(ABC):
    """!
    @class BaseSection
    @brief Abstract base class for all fiber cross-sections.
    """
    @abstractmethod
    def generate_points(self, num_points: int) -> np.ndarray:
        """!
        @brief Generates boundary points in local (u, v) coordinates.
        @param num_points Number of points to sample along the perimeter.
        @return Array of shape (num_points, 3) where the last column is zero.
        """
        pass
    
    @abstractmethod
    def contains_point(self, u: float, v: float) -> bool:
        """!
        @brief Checks if a local 2D point is inside the section.
        @param u Local X coordinate.
        @param v Local Y coordinate.
        @return True if the point is inside or on the boundary.
        """
        pass

    @abstractmethod
    def get_area(self) -> float:
        """!
        @brief Calculates the surface area of the section.
        @return The area as a float.
        """
        pass

class SuperEllipticalSection(BaseSection):
    """!
    @class SuperEllipticalSection
    @brief Represents a Lamé curve (super-ellipse).
    @details Defined by |x/a|^n + |y/b|^n = 1.
    """
    def __init__(self, a: float, b: float, n: float = 2.5):
        """!
        @brief Constructor for SuperEllipticalSection.
        @param a Semi-major axis.
        @param b Semi-minor axis.
        @param n Shape exponent (2.0 = ellipse, >2.0 = squircle/rectangular).
        """
        self.a = a 
        self.b = b 
        self.n = n 
        
    def generate_points(self, num_points: int) -> np.ndarray:
        """!
        @copybrief BaseSection::generate_points
        """
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
        
        # Signed parametric formula to preserve the quadrants
        c = np.cos(theta)
        s = np.sin(theta)
        
        x = self.a * np.sign(c) * (np.abs(c) ** (2 / self.n))
        y = self.b * np.sign(s) * (np.abs(s) ** (2 / self.n))
        
        return np.column_stack([x, y, np.zeros(num_points)])

    def contains_point(self, u: float, v: float) -> bool:
        """!
        @copybrief BaseSection::contains_point
        """
        # Implicit equation: |x/a|^n + |y/b|^n <= 1
        return (abs(u)/self.a)**self.n + (abs(v)/self.b)**self.n <= 1.0

    def get_area(self) -> float:
        """!
        @copybrief BaseSection::get_area
        """
        from scipy.special import gamma
        return 4 * self.a * self.b * (gamma(1 + 1/self.n)**2) / gamma(1 + 2/self.n)

class CircularSection(SuperEllipticalSection):
    """!
    @class CircularSection
    @brief Specialized SuperEllipticalSection for circles (n=2, a=b=radius).
    """
    def __init__(self, radius: float):
        """!
        @brief Constructor for CircularSection.
        @param radius The radius of the circle.
        """
        super().__init__(radius, radius, 2.0)

def create_section(type_str: str, **kwargs):
    """!
    @brief Factory function to create a section object.
    @param type_str Type of section ('circular' or 'superelliptical').
    @param kwargs Parameters for the section (radius, major_radius, minor_radius, exponent).
    @return An instance of a class inheriting from BaseSection.
    """
    # Fall back to a base radius value if needed
    r_default = kwargs.get('radius', 1.0)

    if type_str == 'circular':
        return CircularSection(r_default)

    elif type_str == 'superelliptical':
        # Extract the parameters cleanly
        a = kwargs.get('major_radius', r_default)
        # If minor_radius is not provided, use 70% of a (consistent arbitrary value)
        b = kwargs.get('minor_radius', a * 0.7)
        n = kwargs.get('exponent', 2.5)
        return SuperEllipticalSection(a, b, n)

    raise ValueError(f"Unknown section type: {type_str}")