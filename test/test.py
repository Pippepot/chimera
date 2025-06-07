import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chimera.nodes import *
from chimera.graph import parse_ast
from chimera.helpers import DEBUG
from chimera import renderer, compiler
import unittest, re

DEBUG.value = 0

class Test(unittest.TestCase): 
  def test_print_array(self):
    self.assert_program(Array([1, 2]), '[1, 2]')

  def test_print_multi_array(self):
    self.assert_program(Array([[1, 2],[3, 4]]), '[[1, 2], [3, 4]]')

  def test_add_array(self):
    self.assert_program(Array([1, 2]) + 3, '[4, 5]')

  def test_broadcast(self):
    self.assert_program(Array([[1,2,3],[4,5,6]]) * Array([5,2,10]), '[[5, 4, 30], [20, 10, 60]]')

  def test_reshape(self):
    self.assert_program(Reshape(Array([[1,2,3], [4,5,6]]), (1, 3, 2)), '[[[1, 2], [3, 4], [5, 6]]]')

  def test_index_propagation(self):
    self.assert_program(Index(Array([15, 20]) * Index(Array([40, 50]), 1) + 15, 0), '765')

  def test_slice_simple(self):
    self.assert_program(Index(Array([1, 2, 3]), Slice(1, 3)), '[2, 3]')

  def test_slice(self): 
    self.assert_program(Index(Array([[1,2,3],[4,5,6]]), (1, Slice(0, 3, 2))), '[4, 6]')

  def test_permute(self): 
    self.assert_program(Permute(Array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]), (2, 0, 1)), '[[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]')
  
  def assert_program(self, ast, truth):
    strip_ws = lambda s: re.sub(r"\s+", "", s)
    return self.assertEqual(strip_ws(compiler.compile(*renderer.render(parse_ast(Debug(ast))))), strip_ws(truth))

if __name__ == '__main__':
    unittest.main()