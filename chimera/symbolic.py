from chimera.graph import PatternMatcher, Pat
from chimera.nodes import *
from typing import Callable
import operator, math

def cdiv(x:int, y:int) -> int: return abs(x)//abs(y)*(1,-1)[x*y<0] if y != 0 else 0
def cmod(x:int, y:int) -> int: return x-cdiv(x,y)*y
def safe_exp2(x):
  try: return 2 ** x
  except OverflowError: return math.inf
def safe_pow(x, y):
  try: return math.nan if isinstance(p:=pow(x, y), complex) else p
  except ZeroDivisionError: return math.inf
  except ValueError: return math.inf if x > 0 else -math.inf

python_alu: dict[str, Callable]  = {
  Ops.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan, Ops.EXP2: safe_exp2,
  Ops.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, Ops.SIN: lambda x: math.sin(x) if not math.isinf(x) else math.nan,
  Ops.ADD:operator.add, Ops.SUB:operator.sub, Ops.MUL:operator.mul, Ops.DIV:cdiv, Ops.POW: safe_pow,
  Ops.SHL:operator.lshift, Ops.SHR:operator.rshift, Ops.MOD:cmod, Ops.MAX:max,
  Ops.AND:operator.and_, Ops.OR:operator.or_, Ops.XOR:operator.xor,
  Ops.CMPLT:operator.lt, Ops.CMPNE:operator.lshift}

def execute_alu(op:str, operands):
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