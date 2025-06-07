from chimera.graph import PatternMatcher, Pat
from chimera.nodes import *
from typing import Callable
import operator

python_alu: dict[str, Callable]  = {
  "+":operator.add, "-":operator.sub, "*":operator.mul, "/":operator.truediv, "%":operator.mod, "max":max}

def execute_alu(op:str, operands):
  if op == '/' and all(isinstance(o, int) for o in operands): return Const(operands[0] // operands[1])
  return Const(python_alu[op](*operands))

symbolic = PatternMatcher([
  # Constant folding
  (Pat(BinaryOp, name="op", sources=(Pat(Const), Pat(Const))), lambda op: execute_alu(op.op, [x.value for x in op.sources])),

  # Move constants to the end
  (Pat.cvar("x1") + Pat.var("x2"), lambda x1,x2: x2 + x1),
  (Pat.cvar("x1") * Pat.var("x2"), lambda x1,x2: x2 * x1),

  (Pat(Loop, predicate=lambda x: isinstance(x.idx.data, Const) and isinstance(x.stop, Const) and x.stop.value - x.idx.data.value == 1, name="x"),
   lambda x: x.scope),

  (Pat.var("x") + 0, lambda x: x), # x+0 -> x
  (Pat.var("x") * 1, lambda x: x), # x*1 -> x
  (Pat() * 0, lambda: Const(0)), # x*0 -> 0
  (Pat.var("x") / Pat.var("x"), lambda: Const(1)), # x/x -> 1

  (Pat.var("y") / Pat.var("x") * Pat.var("x"), lambda y, x: y), # (y/x)*x -> y
  (Pat.var("y") * Pat.var("x") / Pat.var("x"), lambda y, x: y), # (y*x)/x -> y
])