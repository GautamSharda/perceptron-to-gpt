import unittest
import math
from gradient_descent import Value, UnidimensionalNeuron

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
        # sgd
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

if __name__ == "__main__":
    unittest.main()