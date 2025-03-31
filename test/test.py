import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chimera.graph import Print, BinaryOp, Index, Array, Const, parse_ast
from chimera import renderer, compiler
import unittest

class Test(unittest.TestCase): 
  def test_print_array(self):
    ast = Print(Array([1, 2]))
    self.assertEqual(self.parse(ast), '1\n2\n')

  def test_print_multi_array(self):
    ast = Print(Array([[1, 2],[3, 4]]))
    self.assertEqual(self.parse(ast), '1\n2\n3\n4\n')

  def test_add_array(self):
    ast = Print(
      BinaryOp(
        op='+',
        left=Array([1, 2]),
        right=Const(3))
      )
    self.assertEqual(self.parse(ast), '4\n5\n')

  def test_index_propagation(self):
    ast = Print(
      Index(
        BinaryOp(
          op='+',
          left=BinaryOp(
            op='*',
            left=Array([15, 20]), 
            right=Index(Array([40, 50]), Const(1))), 
          right=Const(15)),
        Const(0)))
    self.assertEqual(int(self.parse(ast)), 765)

  def parse(self, ast): return compiler.compile(*renderer.render(parse_ast(ast)))

if __name__ == '__main__':
    unittest.main()