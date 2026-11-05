"""numericalhandles.dtypes: DTYPE and as_variable."""

import unittest

import numpy as np
import tensorflow as tf

from tensorquant.numericalhandles.dtypes import DTYPE, as_variable


class TestAsVariable(unittest.TestCase):
    def test_dtype_is_float64(self):
        self.assertEqual(DTYPE, tf.float64)

    def test_reuses_existing_variable(self):
        v = tf.Variable(1.5, dtype=tf.float64)
        self.assertIs(as_variable(v), v)

    def test_casts_float32_variable_and_tensor(self):
        for value in (tf.Variable(1.5, dtype=tf.float32), tf.constant(1.5, dtype=tf.float32)):
            out = as_variable(value)
            self.assertIsInstance(out, tf.Variable)
            self.assertEqual(out.dtype, DTYPE)
            self.assertEqual(float(out), 1.5)

    def test_python_data_has_no_float32_round_trip(self):
        x = 0.1  # not exactly representable: a float32 round trip would change it
        self.assertEqual(float(as_variable(x)), x)
        m = as_variable([[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(m.dtype, DTYPE)
        np.testing.assert_array_equal(m.numpy(), np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_int_and_numpy_inputs(self):
        self.assertEqual(float(as_variable(5000)), 5000.0)
        self.assertEqual(as_variable(np.float32(2.0)).dtype, DTYPE)

    def test_name(self):
        self.assertTrue(as_variable(1.0, name="EQ_EUR_X_SPOT").name.startswith("EQ_EUR_X_SPOT"))


if __name__ == "__main__":
    unittest.main()
