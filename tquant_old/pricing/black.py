import tensorflow as tf 
import tensorflow_probability as tfp
from ..interface.pricer import Pricer 

class BlackScholesPricing(Pricer):
    
    def __init__(self, option, spot_price, risk_free_rate, volatility):
        self.option = option
        self.spot_price = tf.Variable(spot_price, dtype=tf.float32)
        self.risk_free_rate = tf.Variable(risk_free_rate, dtype=tf.float32)
        self.volatility = tf.Variable(volatility, dtype=tf.float32)

    def price(self):
        d1 = (tf.math.log(self.spot_price / self.option.strike_price) + (
            self.risk_free_rate + (self.volatility ** 2) / 2) * self.option.maturity) / (self.volatility * tf.sqrt(self.option.maturity))
        d2 = d1 - self.volatility * tf.sqrt(self.option.maturity)
        dist = tfp.distributions.Normal(0,1)
        call_option_price = self.spot_price * dist.cdf(d1) - self.option.strike_price * tf.exp(-self.risk_free_rate * self.option.maturity) * dist.cdf(d2)

        return call_option_price if self.option.option_type == "Call" else call_option_price - (self.spot_price - self.option.strike_price)*tf.exp(-self.risk_free_rate * self.option.maturity)

    def price_aad(self):
        with tf.GradientTape() as tape:
            tape.watch(self.spot_price)
            tape.watch(self.risk_free_rate)
            tape.watch(self.volatility)

            option_price = self.calculate_option_price()

        aad_greeks = tape.gradient(option_price, [self.spot_price, self.volatility, self.risk_free_rate ])
        #vega = tape.gradient(option_price, self.volatility)

        return option_price, aad_greeks