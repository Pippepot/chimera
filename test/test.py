import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chimera import main
import unittest

class Test(unittest.TestCase):
    def test_add(self):
        result = main.run('print 1 + 2')
        self.assertEqual(int(result), 3)

    def test_operator_precedence(self):
        result = main.run('print 1 + 2 * 3 - 4 / 2')
        self.assertEqual(int(result), 5)

    def test_variable(self):
        result = main.run('a = 1\tprint a')
        self.assertEqual(int(result), 1)

    def test_variable_expression(self):
        result = main.run('a = 2 * 4\tprint a')
        self.assertEqual(int(result), 8)

    def test_variable_name_with_special(self):
        result = main.run('a?3g = 1\tprint a?3g')
        self.assertEqual(int(result), 1)

if __name__ == '__main__':
    unittest.main()