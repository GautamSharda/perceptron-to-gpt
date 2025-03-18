import unittest
import math
from nn import Value, UnidimensionalNeuron

class TestValue(unittest.TestCase):
    def test_init(self):
        v = Value(5.0)
        self.assertEqual(v.value, 5.0)
        self.assertEqual(v.gradient, 0.0)
    
    def test_add(self):
        v1 = Value(5.0)
        v2 = Value(3.0)
        result = v1 + v2
        self.assertEqual(result.value, 8.0)
    
    def test_mul(self):
        v1 = Value(5.0)
        v2 = Value(3.0)
        result = v1 * v2
        self.assertEqual(result.value, 15.0)
    
    def test_sub(self):
        v1 = Value(5.0)
        v2 = Value(3.0)
        result = v1 - v2
        self.assertEqual(result.value, 2.0)
    
    def test_div(self):
        v1 = Value(6.0)
        v2 = Value(3.0)
        result = v1 / v2
        self.assertEqual(result.value, 2.0)

class TestUnidimensionalNeuron(unittest.TestCase):
    def test_init(self):
        neuron = UnidimensionalNeuron()
        self.assertIsInstance(neuron.weight, Value)
        self.assertIsInstance(neuron.bias, Value)
        self.assertIsInstance(neuron.learning_rate, Value)
        self.assertEqual(neuron.learning_rate.value, 0.1)
    
    def test_forward(self):
        neuron = UnidimensionalNeuron()
        # Set specific values for testing
        neuron.weight = Value(2.0)
        neuron.bias = Value(1.0)
        
        result = neuron.forward(3.0)
        self.assertEqual(result.value, 7.0)  # 2.0 * 3.0 + 1.0 = 7.0
    
    def test_forward_invalid_input(self):
        neuron = UnidimensionalNeuron()
        with self.assertRaises(ValueError):
            neuron.forward("invalid")
    
    def test_descend(self):
        neuron = UnidimensionalNeuron()
        neuron.weight = Value(2.0)
        neuron.weight.gradient = 0.5
        neuron.learning_rate = Value(0.1)
        
        neuron.descend()
        self.assertEqual(neuron.weight.value, 1.95)  # 2.0 - 0.1 * 0.5 = 1.95

if __name__ == "__main__":
    unittest.main()