# bias.py (the class part)
from surprise import BaselineOnly

from .base import SurpriseModel


class BiasModel(SurpriseModel):
    """f(u,i) = alpha + beta_u + beta_i, fit by ALS (Surprise BaselineOnly)."""

    def __init__(self, reg_u: float, reg_i: float, n_epochs: int = 25,
                 scale: tuple[float, float] = (1, 5)) -> None:
        super().__init__(scale)
        self.reg_u, self.reg_i, self.n_epochs = reg_u, reg_i, n_epochs

    def _make_algo(self) -> BaselineOnly:
        return BaselineOnly(bsl_options={"method": "als", "reg_u": self.reg_u,
                                         "reg_i": self.reg_i, "n_epochs": self.n_epochs},
                            verbose=False)
