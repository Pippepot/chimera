import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chimera.nodes import *
from chimera.graph import parse_ast
from chimera.helpers import DEBUG, TRACK_REWRITES, LOG_REWRITE_FAILURES
from chimera import compiler, renderer
import time

"""
TODO:
Array features:
  Reduce op, symbolic indexing
    SLICE
    PAD ? CAT ?
    FLIP ?
Language features
  function, exp, log, sqrt, sin, cos, abs, input, min, max comparisons, bool
  branch, strings, lists, tuples, dicts, sets, enums
Graph features:
  Symbolic rewrites, constant folding, loop unrolling, loop fusion, factorize common expressions
Debugging
  Track rewrites
"""

DEBUG.value = 2
TRACK_REWRITES.value = 0
LOG_REWRITE_FAILURES.value = 1

def main():
  
  ast = [
    # Debug(Reshape(Array([[1,2,3],[4,5,6]]) * Array([5,2,10]), (1,3,1,2)) + 3), 
    # Debug(Reshape(Array([[1,2,3],[4,5,6]]) * Array([5,2,10]), (1,3,1,2)) + 3),
    Debug(Index(Array([[1,2,3],[4,5,6]]), (1, Slice(0, 3, 1)))),
    # Debug(Reshape(Array([[1,2,3], [4,5,6]]), (1, 3, 2))),
    # Debug(Index(Expand(Array([[1,2,3],[4,5,6]]), (2, 2, 3)), (1,))),
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