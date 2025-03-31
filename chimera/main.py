import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chimera.graph import Print, BinaryOp, Index, Array, Const, Loop, Assign, Assign, Var, parse_ast
from chimera import compiler, renderer
import time

def main():
  ast = [
    Print(BinaryOp('*', Array([[[1, 2, 3],[3, 4, 5]],[[6, 7, 8],[9, 10, 11]]]), Const(1)))
  ]
  
  for node in ast: node.print_tree()
  compile_timer = time.perf_counter()
  procedure = parse_ast(ast)
  for p in procedure: print(p)
  code, functions = renderer.render(procedure)
  print(f"Chimera compile\t{time.perf_counter() - compile_timer:.4f}ms")
  result = compiler.compile(code, functions)
  print(result)

if __name__ == "__main__":
  main()