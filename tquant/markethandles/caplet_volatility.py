import numpy as np

from tquant.markethandles.bicubic_spline_interpolator import BiCubicSplineInterpolator
from tquant.markethandles.bilinear_interpolator import BilinearInterpolator
from tquant.markethandles.linear_cubic_interpolator import LinearCubicInterpolator


class CapletVolatility:
    def __init__(self,
                 volatility_matrix: np.ndarray,
                 maturity_interp: str = "LINEAR",
                 strike_interp: str = "LINEAR"):
        # min_size = 2x2
        if volatility_matrix.shape[0] < 2 or volatility_matrix.shape[1] < 2:
            raise ValueError("Wrong matrix specified")
        terms = volatility_matrix[1:, 0]
        strikes = volatility_matrix[0, 1:]
        vols = volatility_matrix[1:, 1:]
        self.transpose = False
        if maturity_interp == "LINEAR" and strike_interp == "LINEAR":
            self.interpolator = BilinearInterpolator(terms, strikes, vols)
        elif maturity_interp == "LINEAR" and strike_interp == "CUBIC_SPLINE":
            self.interpolator = LinearCubicInterpolator(terms, strikes, vols)
        elif maturity_interp == "CUBIC_SPLINE" and strike_interp == "LINEAR":
            self.interpolator = LinearCubicInterpolator(strikes, terms, vols.T)
            self.transpose = True
        elif maturity_interp == "CUBIC_SPLINE" and strike_interp == "CUBIC_SPLINE":
            self.interpolator = BiCubicSplineInterpolator(terms, strikes, vols)
        else:
            raise ValueError("Interpolation not supported")

    def interpolate(self, maturity: float, strike: float):
        if self.transpose:
            return self.interpolator.interpolate(strike, maturity)
        else:
            return self.interpolator.interpolate(maturity, strike)


if __name__ == "__main__":
    vol_matrix = np.array([[0.0, 100, 200, 300],
                           [1, 0.2, 0.3, 0.4],
                           [2, 0.5, 0.6, 0.7]])
    volatility_interpolator = CapletVolatility(vol_matrix, "LINEAR", "CUBIC_SPLINE")

    v1 = volatility_interpolator.interpolate(1, 100)
    v2 = volatility_interpolator.interpolate(1, 200)
    v3 = volatility_interpolator.interpolate(1, 300)
    v4 = volatility_interpolator.interpolate(2, 100)
    v5 = volatility_interpolator.interpolate(2, 200)
    v6 = volatility_interpolator.interpolate(2, 300)
    print(v1)
    print(v2)
    print(v3)
    print(v4)
    print(v5)
    print(v6)
    v7 = volatility_interpolator.interpolate(1, 150)
    print(v7)
    v8 = volatility_interpolator.interpolate(1.5, 100)
    print(v8)
    v9 = volatility_interpolator.interpolate(1.5, 150)
    print(v9)
