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

Redo graph rewrites?
  tinygrad uops are immutable. All changes to uops are through entire graph rewrites
  NEVERMIND tinygrad uops outside graph rewrites
  not all uops have views (only loads?)
  All uops have the same structure (op, dtype, src, args) which makes cloning a uop possible
  Rewrite nodes in tinygrad style??

  Rewrite rules
    Bottom up
      Can change shape/view on self
      Cannot change shape/view of children
    Top down
      Cannot change shape/view of self
      Can change shape/view of children

  We go top down
  Indices can propagate down, parent shapes should not depend on child shapes during rewrite
"""

DEBUG.value = 2

def main():
  ast = [
    Print(BinaryOp('*', Array([[[1, 2],[3, 4]],[[5, 6],[7, 8]]]), Array([2])))
  ]
  
  if DEBUG:
    compile_timer = time.perf_counter()
  procedure = parse_ast(ast)
  code, functions = renderer.render(procedure)
  if DEBUG:
    print(f"Chimera compile\t{time.perf_counter() - compile_timer:.4f}s")
  result = compiler.compile(code, functions)
  print(result)

if __name__ == "__main__":
  main()