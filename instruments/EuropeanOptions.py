from abc import ABC, abstractmethod
import tensorflow as tf


class EuropeanOptionInstrument(ABC):
    def __init__(self, strike_price, maturity, option_type):
        self.strike_price = tf.Variable(strike_price, dtype=tf.float32)
        self.maturity = tf.Variable(maturity, dtype=tf.float32)
        self.option_type = option_type  # "Call" or "Put"

    @abstractmethod
    def calculate_payoff(self, market_price):
        pass

    @abstractmethod
    def calculate_intrinsic_value(self, market_price):
        pass

    @abstractmethod
    def calculate_time_value(self, market_price):
        pass

    def display_information(self):
        print(f"Strike Price: {self.strike_price}")
        print(f"Maturity: {self.maturity} years")
        print(f"Option Type: {self.option_type}")

    @property
    @abstractmethod
    def is_exercisable(self):
        pass



class EuropeanCallOption(EuropeanOptionInstrument):
    def calculate_payoff(self, market_price):
        return max(market_price - self.strike_price, 0)

    def calculate_intrinsic_value(self, market_price):
        return max(market_price - self.strike_price, 0)

    def calculate_time_value(self, market_price):
        intrinsic_value = self.calculate_intrinsic_value(market_price)
        return max(0, market_price - intrinsic_value)

    @property
    def is_exercisable(self):
        return True if self.maturity > 0 else False