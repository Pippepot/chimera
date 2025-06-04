import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chimera.nodes import *
from chimera.graph import parse_ast
from chimera.helpers import DEBUG, TRACK_REWRITES
from chimera import compiler, renderer
import time

"""
TODO:
Array features:
  Reduce op, symbolic indexing, range indexing, broadcasting
Language features
  function, exp, log, sqrt, sin, cos, abs, input, min, max comparisons, bool
  branch, strings, lists, tuples, dicts, sets, enums
Graph features:
  Symbolic rewrites, constant folding, loop unrolling, loop fusion, factorize common expressions
Debugging
  Track rewrites
"""

DEBUG.value = 0
TRACK_REWRITES.value = 0

def main():
  ast = [
    # Print(Reshape(Array([[1,2,3],[4,5,6]]) * Array([5,2,10]), (1,3,1,2)) + 3),
    # Print(Array([[1,2,3],[4,5,6]]) * Array([5,2,10])),
    # Debug(Expand(Array([5,2,10]), View.create((2,3)))),
    Debug(Array([[1,2,3],[4,5,6]])+Array([[1,2,3],[4,5,6]])),
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