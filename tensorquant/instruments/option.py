from abc import ABC, abstractmethod
from ..markethandles.utils import Currency, OptionType, ExerciseType
from .product import Product

from datetime import date
import tensorflow as tf


class Option(Product, ABC):

    def __init__(
        self,
        ccy: Currency,
        start_date: date,
        end_date: date,
        option_type: OptionType,
        strike: float | list[float],
        exercise_type: ExerciseType,
        underlying: str = "DEFAULT",
    ):
        super().__init__(ccy, start_date, end_date)
        self._option_type = option_type
        self._strike = tf.Variable(strike, dtype=tf.float32)
        self._underlying = underlying
        self._exercise_type = exercise_type
        self._implied_volatility = None
        self._forward = None

    @property
    def option_type(self):
        return self._option_type

    @property
    def strike(self):
        return self._strike

    @property
    def underlying(self):
        return self._underlying

    @property
    def exercise_type(self):
        return self._exercise_type

    @property
    def implied_volatility(self) -> tf.Variable:
        return self._implied_volatility

    @implied_volatility.setter
    def implied_volatility(self, value: tf.Variable):
        self._implied_volatility = value

    @property
    def forward(self):
        if self._forward is None:
            raise ValueError("forward is not available: price the option first")
        return self._forward

    @forward.setter
    def forward(self, value):
        self._forward = value


class VanillaOption(Option):

    def __init__(
        self,
        ccy: Currency,
        start_date: date,
        end_date: date,
        option_type,
        strike,
        underlying: str = "DEFAULT",
        exercise_type: ExerciseType = ExerciseType.European,
    ):
        super().__init__(
            ccy,
            start_date,
            end_date,
            option_type,
            strike,
            exercise_type,
            underlying,
        )
        self._delta = None
        self._gamma = None
        self._theta = None
        self._vega = None
        self._rho = None
