import unittest
import math
from gradient_descent import Value, UnidimensionalNeuron
import sympy as sp
import numpy as np

class TestValue(unittest.TestCase):
    # ok so here L = |y_hat - y|, where y_hat = mx + b, you have L = |mx + b - y|, 
    # so dm/dL = (1 if mx + b - y >= 0 else -1)*x and db/dL = (1 if mx + b - y >= 0 else -1)*1.0
    # In the test case that is 1*2 and 1*1 respectively
    def test_sgd(self):
        # sgd
        batch_loss = Value(0.0)
        m = Value(20.0)
        x_1 = Value(2)
        b = Value(10)
        y_hat_1 = m*x_1 + b
        y_1 = Value(2.0)
        loss_1 = abs(y_hat_1 - y_1)
        loss_1.backward()
        self.assertEqual(m.gradient, 2.0)
        self.assertEqual(b.gradient, 1.0)
        batch_loss += loss_1
        # accumulate -- now, you still have the same formula for dm/dL and db/dL, which gives 1*3 and 1*1
        # the weight and bias gradients are accumulated to sum to 2 + 3 = 5 and 1 + 1 = 2
        x_2 = Value(3)
        y_2 = Value(3.0)
        y_hat_2 = m*x_2 + b
        loss_2 = abs(y_hat_2 - y_1)
        loss_2.backward()
        self.assertEqual(m.gradient, 5.0)
        self.assertEqual(b.gradient, 2)
        batch_loss += loss_2
    
    def test_bgd(self):
        # bgd
        batch_loss = Value(0.0)
        m = Value(20.0)
        x_1 = Value(2)
        b = Value(10)
        y_hat_1 = m*x_1 + b
        y_1 = Value(2.0)
        loss_1 = abs(y_hat_1 - y_1)
        batch_loss += loss_1
        x_2 = Value(3)
        y_2 = Value(3.0)
        y_hat_2 = m*x_2 + b
        loss_2 = abs(y_hat_2 - y_2)
        batch_loss += loss_2
        # now compute gradients
        avg_batch_loss = batch_loss / Value(2)
        avg_batch_loss.backward()
        self.assertEqual(m.gradient, 2.5)
        self.assertEqual(b.gradient, 1.0)

    def test_bgd_symbolic_verification(self):
        """
        Verifies the gradients calculated by Value.backward() for a BGD scenario
        using SymPy for symbolic differentiation via lambdify.
        """
        #compute gradients using my Value class
        m_val = Value(20.0)
        b_val = Value(10.0)
        x1_val = Value(2.0)
        y1_val = Value(2.0)
        x2_val = Value(3.0)
        y2_val = Value(3.0)
        batch_loss_val = Value(0.0)
        y_hat_1_val = m_val * x1_val + b_val
        loss_1_val = abs(y_hat_1_val - y1_val)
        batch_loss_val += loss_1_val
        y_hat_2_val = m_val * x2_val + b_val
        loss_2_val = abs(y_hat_2_val - y2_val)
        batch_loss_val += loss_2_val
        avg_batch_loss_val = batch_loss_val / Value(2.0)
        avg_batch_loss_val.backward()

        # Set up gradients computation symbolically using SymPy
        m_sym, b_sym, x1_sym, y1_sym, x2_sym, y2_sym = sp.symbols('m b x_1 y_1 x_2 y_2', real=True)

        y_hat_1_sym = m_sym * x1_sym + b_sym
        loss_1_sym = sp.Abs(y_hat_1_sym - y1_sym)
        y_hat_2_sym = m_sym * x2_sym + b_sym
        loss_2_sym = sp.Abs(y_hat_2_sym - y2_sym)
        batch_loss_sym = loss_1_sym + loss_2_sym
        avg_batch_loss_sym = batch_loss_sym / 2

        grad_m_sym = sp.diff(avg_batch_loss_sym, m_sym)
        grad_b_sym = sp.diff(avg_batch_loss_sym, b_sym)

        #this computes symbolic gradients numerically using lambdify
        variables = [m_sym, b_sym, x1_sym, y1_sym, x2_sym, y2_sym]

        grad_m_func = sp.lambdify(variables, grad_m_sym, modules=['numpy'])
        grad_b_func = sp.lambdify(variables, grad_b_sym, modules=['numpy'])

        numeric_values = (
            m_val.value, b_val.value,
            x1_val.value, y1_val.value,
            x2_val.value, y2_val.value
        )

        numeric_grad_m = grad_m_func(*numeric_values)
        numeric_grad_b = grad_b_func(*numeric_values)

        self.assertAlmostEqual(numeric_grad_m, m_val.gradient, places=7)
        self.assertAlmostEqual(numeric_grad_b, b_val.gradient, places=7)

if __name__ == "__main__":
    unittest.main()