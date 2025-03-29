import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chimera.graph import Print, BinaryOp, Index, Array, Const, Loop, Assign, Call, Function, Assign, Var, parse_ast
from chimera import compiler, renderer
import time

def main():
  ast = [
    Print(
      Index(
        BinaryOp(
          op='+',
          left=BinaryOp(
            op='*',
            left=Array([15, 20]), 
            right=Array([40, 50])), 
          right=Const(15)),
        Const(0)))
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