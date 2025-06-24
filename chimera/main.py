import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chimera.nodes import *
from chimera.graph import parse_ast
from chimera.helpers import DEBUG, TRACK_REWRITES, LOG_REWRITE_FAILURES, LOG_SHAPES
from chimera import compiler, renderer
import time

"""
TODO:
Array features:
  Reduce op, symbolic indexing
    PAD ? CAT ?
  Array creation
    Ones, Full,
Language features
  function, exp, log, sqrt, sin, cos, comparisons, bool
  branch, strings, lists
Graph features:
  loop unrolling, loop fusion, factorize common expressions
"""

DEBUG.value = 2
TRACK_REWRITES.value = 0
LOG_REWRITE_FAILURES.value = 1
LOG_SHAPES.value = 1

def main():

  # arg = Var(Array([0, 0]))
  # func = Function(arg * 7, arg)
  ast = [
    Debug(Array([[1,2,3],[4,5,6]]) + 60)
    # Debug((Array([[1,2,3],[4,5,6]]) + 60))
    # Debug(Array([[1,2,3],[4,5,6]]) * Array([5,2,10]))
    # Debug(Array([True, False, True]).where(Array([1,2,3]), Array([10])))
    # Debug(Index(Array([[1,2,3],[4,5,6]]), (1,Slice(0, 2))))
    # Debug(Branch(Const(1) < Const(2), Permute(Reshape(Array([[1,2,3],[4,5,6]]) * 5, (1, 3, 2)), (2, 0, 1)), Const(1))),
    # Debug(Const(1) + 4 << 3),
    # Debug((Array([False, True, False])).where(Array([1,2,3]), Array([4]))),
    # Branch(Const(True), Debug(Array([[1,2,3],[4,5,6]]) * 5), Debug(Const(123))),
    # Debug(Reshape(Array([[1,2,3],[4,5,6]]) * Array([5,2,10]), (1,3,1,2)) + 3),
    # Debug(Index(Array([[1,2,3],[4,5,6]]), (1, Slice(0, 3, 1)))),
    # Debug(Reshape(Array([[1,2,3], [4,5,6]]), (1, 3, 2)))
    # Debug(Permute(Array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]), (2, 0, 1)))
    # Debug(Index(Expand(Array([[1,2,3],[4,5,6]]), (2, 2, 3)), (1,))),
    # func,
    # Debug(Reduce(Array([[1,2,3],[4,5,6]]), func, 1))
  ]

  if DEBUG:
    compile_timer = time.perf_counter()
  procedure = parse_ast(ast)
  code, functions = renderer.render(procedure)
  if DEBUG:
    print(f"Chimera compile\t{(time.perf_counter() - compile_timer) * 1000:.1f}ms")
  result = compiler.compile(code, functions)
  print(result)

if __name__ == "__main__":
  main()