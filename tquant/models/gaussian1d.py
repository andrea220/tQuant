from .hullwhite import HullWhiteProcess
import numpy as np
import math
import tensorflow as tf


class GaussianRateKernel1D:
    def __init__(self, process: HullWhiteProcess) -> None:
        self.process = process

    def rate_tensor(self, 
                    nPaths: int,
                    last_grid_time: float, 
                    time_steps: float,
                    time_grid: np.array = None) -> tuple[tf.Tensor, np.array]:
        """ 
        This function returns a path drawn from
        the Gaussian distribution of the conditional short rate.
        
        Parameters:
        last_grid_time : time span in years.
        time_steps : number of steps in the discretized path.
        
        Returns:
        times : array containing the times points in years.
        rt_path : array containing the short rate points.
        """
        if time_grid is None:
            time_grid = np.linspace(start=0.0, stop=last_grid_time, num=time_steps, retstep=False)
        W = tf.random.normal(shape=(nPaths, len(time_grid)), mean=0, stddev=1, dtype= tf.float64)
        pillars_dfs = self.process.market_curve.pillars

        rates = []
        zb_tensor = []
        rates.append(tf.fill((nPaths,),
                             value= self.process.market_curve.inst_fwd(0)))

        for i in range(1, len(time_grid)):

            s = time_grid[i-1]
            t = time_grid[i]
            dt = t-s
            cond_var = self.process.sigma**2 / (2*self.process.mean_rev) * (1 - math.exp(-2*self.process.mean_rev*dt))
            std_dev = math.sqrt(cond_var)

            a_t = self.process.alpha(t)
            a_s = self.process.alpha(s)
            decay = math.exp(- self.process.mean_rev * dt)
            E_rt = rates[i-1] * decay + a_t - a_s * decay
            r_t = E_rt + std_dev*W[:,i]
            rates.append(r_t)
            zb_curve_i = tf.Variable([self.process.zero_bond(t, t + pillar, r_t) for pillar in pillars_dfs])
            zb_tensor.append(zb_curve_i)
        zb_tensor = tf.Variable(zb_tensor)
        rates = tf.Variable(rates)

        return rates, zb_tensor, time_grid
