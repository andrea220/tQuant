from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
from instruments.simpleeuropeanoption import *


class EuropeanOptionPricing(ABC):
    @abstractmethod
    def calculate_option_price(self, option: EuropeanOptionInstrument):
        pass

    @abstractmethod
    def calculate_aad(self, option: EuropeanOptionInstrument):
        pass



class BlackScholesPricing(EuropeanOptionPricing):
    
    def __init__(self, option, spot_price, risk_free_rate, volatility):
        self.option = option
        self.spot_price = tf.Variable(spot_price, dtype=tf.float32)
        self.risk_free_rate = tf.Variable(risk_free_rate, dtype=tf.float32)
        self.volatility = tf.Variable(volatility, dtype=tf.float32)

    def calculate_option_price(self):
        d1 = (tf.math.log(self.spot_price / self.option.strike_price) + (
            self.risk_free_rate + (self.volatility ** 2) / 2) * self.option.maturity) / (self.volatility * tf.sqrt(self.option.maturity))
        d2 = d1 - self.volatility * tf.sqrt(self.option.maturity)
        dist = tfp.distributions.Normal(0,1)
        call_option_price = self.spot_price * dist.cdf(d1) - self.option.strike_price * tf.exp(-self.risk_free_rate * self.option.maturity) * dist.cdf(d2)

        return call_option_price if self.option.option_type == "Call" else call_option_price - (self.spot_price - self.option.strike_price)

    def calculate_aad(self):
        with tf.GradientTape() as tape:
            tape.watch(self.spot_price)
            tape.watch(self.risk_free_rate)
            tape.watch(self.volatility)

            option_price = self.calculate_option_price()

        aad_greeks = tape.gradient(option_price, [self.spot_price, self.volatility, self.risk_free_rate ])
        #vega = tape.gradient(option_price, self.volatility)

        return option_price, aad_greeks
    
    def calculate_aad_xla(self):
        # Apply XLA compilation to the gradient calculation
        @tf.function(jit_compile=True)
        def compute_gradients():
            with tf.GradientTape() as tape:
                tape.watch(self.spot_price)
                tape.watch(self.risk_free_rate)
                tape.watch(self.volatility)

                option_price = self.calculate_option_price()

            #delta = tape.gradient(option_price, self.spot_price)
            aad_greeks = tape.gradient(option_price, [self.spot_price, self.volatility, self.risk_free_rate ])
            return option_price, aad_greeks

        return compute_gradients()
    

#from instruments.EuropeanOptions import EuropeanOptionInstrument


class EuropeanKernel(EuropeanOptionPricing):
    
    def __init__(self, option, spot_price, mu, sigma, z):
        self.option = option
        self.spot_price = tf.Variable(spot_price, dtype=tf.float32)
        self.mu = tf.Variable(mu, dtype=tf.float32)
        self.sigma = tf.Variable(sigma, dtype=tf.float32)
        self.z = z


    def calculate_option_price(self):
        dt = self.option.maturity / self.z.shape[1]
        dt_sqrt = tf.math.sqrt(dt)
        diffusion = self.sigma * dt_sqrt
        drift = (self.mu - (self.sigma ** 2) / 2)
        gbm = tf.math.exp(drift * dt + diffusion * self.z)
        s_t = self.spot_price * tf.math.cumprod(gbm, axis=1)

        payoff = tf.math.maximum(s_t[:, -1] - self.option.strike_price, 0)
        return tf.exp(-self.mu * self.option.maturity) * tf.reduce_mean(payoff)
    
    def calculate_option_price_xla(self):
        @tf.function(jit_compile = True)
        def compute_price():
            dt = self.option.maturity / self.z.shape[1]
            dt_sqrt = tf.math.sqrt(dt)
            diffusion = self.sigma * dt_sqrt
            drift = (self.mu - (self.sigma ** 2) / 2)
            gbm = tf.math.exp(drift * dt + diffusion * self.z)
            s_t = self.spot_price * tf.math.cumprod(gbm, axis=1)

            payoff = tf.math.maximum(s_t[:, -1] - self.option.strike_price, 0)
            return tf.exp(-self.mu * self.option.maturity) * tf.reduce_mean(payoff)
        return compute_price()


    def calculate_aad(self):
        with tf.GradientTape() as tape:
            option_price = self.calculate_option_price()

        aad_greeks = tape.gradient(option_price, [self.spot_price, self.sigma, self.mu])

        return option_price, aad_greeks
    

    def calculate_aad_xla(self):
        @tf.function(jit_compile=True)
        def compute_gradients():
            with tf.GradientTape() as tape:
                option_price = self.calculate_option_price()

            aad_greeks = tape.gradient(option_price, [self.spot_price, self.sigma, self.mu])

            return option_price, aad_greeks

        return compute_gradients()
