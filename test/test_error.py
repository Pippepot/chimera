import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from chimera import main
from unittest.mock import patch
from io import StringIO

class Test(unittest.TestCase):
    @patch('sys.stdout', new_callable=StringIO)
    def test_undeclared_identifier(self, mock_stdout):
        main.run('print x')
        self.assertIn("Undeclared identifier", mock_stdout.getvalue())

if __name__ == '__main__':
    unittest.main()