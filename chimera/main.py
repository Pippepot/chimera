import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chimera.graph import Print, BinaryOp, Index, Array, Const, Loop, Var, parse_ast
from chimera.helpers import DEBUG
from chimera import compiler, renderer
import time

"""
TODO:
Array features:
  Reduce op, symbolic indexing, range indexing, broadcasting
  better printing, empty array initialization, reshaping
Language features
  function, exp, log, sqrt, sin, cos, abs, input, min, max comparisons, bool
  branch, strings
Pretty print linearized program
"""

DEBUG.value = 2

def main():
  ast = [
    Print(BinaryOp('*', Array([[[1, 2],[3, 4]],[[5, 6],[7, 8]]]), Const(2)))
  ]
  
  if DEBUG:
    compile_timer = time.perf_counter()
  procedure = parse_ast(ast)
  code, functions = renderer.render(procedure)
  if DEBUG:
    print(f"Chimera compile\t{time.perf_counter() - compile_timer:.4f}ms")
  result = compiler.compile(code, functions)
  print(result)

if __name__ == "__main__":
  main()