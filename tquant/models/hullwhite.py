import tensorflow as tf
import math
from tensorflow.python.framework import dtypes
from .stochasticprocess import StochasticProcess
from .ornsteinuhlenbeck import OrnsteinUhlenbeckProcess
from ..markethandles.ircurve import RateCurve

class HullWhiteProcessOld:
    def __init__(self, mean_rev: float, sigma: float, market_curve) -> None:
        self.sigma = tf.Variable(mean_rev, dtype=dtypes.float64)
        self.mean_rev = tf.Variable(sigma, dtype=dtypes.float64)
        self.market_curve = market_curve
        
    
    def alpha(self, t: float) -> float:
        """ 
        This function returns the alpha 
        time-dependent parameter.
        α(t) = f(0, t) + 0.5(σ(1-exp(-kt))/k)^2
        
        Parameters:
        t : reference time in years.
        
        Returns:
        α(t) : deterministic parameter to recover term-rates.
        """
        f = self.market_curve.inst_fwd(t)
        return f + self.sigma**2 / (2 * self.mean_rev**2) * (1 - math.exp(- self.mean_rev*t))**2
        
    def conditional_moments(self, s: float, t: float, r_s: tf.Tensor) -> tuple[tf.Tensor, float]:
        """ 
        This function returns the conditional mean
        and conditional variance of the short rate process 
        given the known value
        at time s <= t.
        E{r(t)|r(s)} = (r(s) - α(s))*exp(-k(t-s)) + α(t)
        Var{r(t)|r(s)} = σ^2[1 - exp(-2k(t-s))]/(2k)
        
        Parameters:
        s : information stopping time in years.
        t : reference time in years.
        r_s : short rate known at time s.
        
        Returns:
        E{r(t)|r(s)} : conditional mean
        Var{r(t)|r(s)} : conditional variance
        """
        dt = t-s
        a_t = self.alpha(t)
        a_s = self.alpha(s)
        decay = math.exp(- self.kappa * dt)
        E_rt = r_s * decay + a_t - a_s * decay
        Var_rt = self.sigma**2 / (2*self.kappa) * (1 - math.exp(-2*self.kappa*dt))
        return E_rt, Var_rt
    
    def A_B(self, S: float, T: float) -> tuple[float, float]:
        """ 
        This function returns the time dependent parameters
        of the ZCB, where S <= T.
        B(S, T) = (1 - exp(-k(T-S)))/k
        A(S, T) = P(0,T)/P(0,S) exp(B(S,T)f(0,S) - 
                    σ^2(exp(-kT)-exp(-kS))^2(exp(2kS)-1)/(4k^3))
        
        Parameters :
        S : future reference time of the ZCB in years.
        T : future reference maturity of the ZCB years.
        
        Returns : 
        A(S, T) : scale factor of the ZCB
        B(S, T) : exponential factor of the ZCB
        """
        f0S = self.market_curve.inst_fwd(S) 
        P0T = self.market_curve.discount(T)
        P0S = self.market_curve.discount(S)

        B = 1 - tf.exp(-self.mean_rev*(T - S))
        B /= self.mean_rev
        
        exponent = self.sigma*(tf.exp(-self.mean_rev*T) - tf.exp(-self.mean_rev*S))
        exponent *= exponent
        exponent *= tf.exp(2*self.mean_rev*S) - 1
        exponent /= -4*(self.mean_rev**3)
        exponent += B*f0S
        A = tf.exp(exponent)*P0T/P0S
        return A, B
    
    def zero_bond(self, S: float, T: float, rs: float) -> float:
        """ 
        This function returns the price of a ZCB
        P(S, T) at future reference time S and maturity T 
        with S <= T.
        
        Parameters :
        S : future reference time of the ZCB in years.
        T : future reference maturity of the ZCB years. 
        
        Returns :
        P(S, T) : ZCB price with maturity T at future date S.
        """
        A, B = self.A_B(S, T)
        return A*tf.exp(-B*rs)
    

class HullWhiteProcess(StochasticProcess):
    def __init__(self,
                 term_structure: RateCurve,
                 a: float,
                 sigma: float):
        self._process = OrnsteinUhlenbeckProcess(mr_speed=a, volatility=sigma, x0=term_structure.inst_fwd(0))
        self._a = tf.Variable(a, dtype=dtypes.float64)
        self._sigma = tf.Variable(sigma, dtype=dtypes.float64)
        self._term_structure = term_structure

    def size(self):
        return self._process.size()
    
    def initial_values(self):
        return self._process.x0
    
    @property
    def a(self):
        return self._a.numpy()

    @property
    def sigma(self):
        return self._sigma.numpy()

    @property
    def x0(self):
        return self._process.x0.numpy()
    
    def drift(self, t, x):
        alpha_drift = self._sigma**2 / (2 * self._a) * (1 - tf.exp(-2 * self._a * t))
        shift = 0.0001
        f = self._term_structure.forward_rate(t, t)
        fup = self._term_structure.forward_rate(t + shift, t + shift)
        f_prime = (fup - f) / shift
        alpha_drift += self._a * f + f_prime
        return self._process.drift(t, x) + alpha_drift

    def diffusion(self, t, x):
        return self._process.diffusion(t, x) # perché diffusion ha (t,x) se sul c++ ritorna sempre volatility_?

    def expectation(self, t0, x0, dt):
        return (self._process.expectation(x0=x0, dt=dt) 
                + self.alpha(t0 + dt) - self.alpha(t0) * tf.exp(-self._a * dt))

    def std_deviation(self, dt, t0=None, x0=None):
        return self._process.std_deviation(dt=dt)

    def variance(self, dt):
        return self._process.variance(dt)

    def alpha(self, t):
        if self.a > 1e-10:
            alfa = (self._sigma / self._a) * (1 - tf.exp(-self._a * t))
        else:
            alfa = self._sigma * t
        alfa = 0.5 * alfa**2
        alfa += self._term_structure.inst_fwd(t)
        return alfa