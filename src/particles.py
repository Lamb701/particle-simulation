from dataclasses import dataclass
import numpy as np


@dataclass
class Particle:
    charge: float          # C
    mass: float            # kg
    position: np.ndarray   # m, shape (3,)
    velocity: np.ndarray   # m/s, shape (3,)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).copy()
        self.velocity = np.asarray(self.velocity, dtype=float).copy()
        if self.position.shape != (3,):
            raise ValueError("position must have shape (3,)")
        if self.velocity.shape != (3,):
            raise ValueError("velocity must have shape (3,)")

    def copy(self) -> "Particle":
        return self.__class__(
            charge=self.charge,
            mass=self.mass,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
        )

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * float(self.velocity @ self.velocity)
