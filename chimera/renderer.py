from __future__ import annotations
from chimera.nodes import *
from chimera.rewrite import PatternMatcher, Pat
from chimera.dtype import dtypes
from chimera.helpers import TRACK_REWRITES, navigate_history

op_rendering: dict = {
  Ops.SQRT: lambda x: f"sqrt({x})", Ops.EXP2: lambda x: f"exp2({x})",
  Ops.LOG2: lambda x: f"log2({x})", Ops.SIN: lambda x: f"sin({x})",
  Ops.ADD: lambda a,b: f"({a}+{b})", Ops.SUB: lambda a,b: f"({a}-{b})",
  Ops.MUL: lambda a,b: f"({a}*{b})", Ops.DIV: lambda a,b: f"({a}/{b})",
  Ops.SHL: lambda a,b: f"({a}<<{b})", Ops.SHR: lambda a,b: f"({a}>>{b})",
  Ops.MOD: lambda a,b: f"({a}%{b})", Ops.MAX: lambda a,b: f"({a}>{b}?{a}:{b})",
  Ops.AND: lambda a,b: f"({a}&{b})", Ops.OR: lambda a,b: f"({a}|{b})", Ops.XOR: lambda a,b: f"({a}^{b})",
  Ops.CMPLT: lambda a,b: f"({a}<{b})", Ops.CMPNE: lambda a,b: f"({a}!={b})",
  Ops.POW: lambda a,b: f"pow({a}, {b})",
}

class NodeGroup:
  Associative = {'+', '*'}
  Terminal = {Debug, Loop}

class TypeGroup:
  Number = {dtypes.int, dtypes.int}

dtype_to_str = {dtypes.int:'int', dtypes.float:'float', dtypes.bool:'char', None:'void'}

def render_assign(ctx, x:Assign):
  # C initializes arrays and pointers with different syntax
  if isinstance(x.var.data, Array):
    return f'{dtype_to_str[x.var.dtype]} {ctx[x.var]}[] = {ctx[x.var.data]};'
  pointer = '*' if x.var.shape != () else ''
  return f'{dtype_to_str[x.var.dtype]}{pointer} {ctx[x.var]} = {ctx[x.var.data]};'
def render_array(data, suffix=''): return f"{{{','.join([f'{v}{suffix}' for v in data])}}}"
def dtype_suffix(dtype) -> str: return 'f' if dtype==dtypes.float else ''
def append_indent(string:str) -> str: return "\n".join(f"  {line}" for line in string.splitlines())
def strip_parens(string:str) -> str: return string[1:-1]

render_patterns = PatternMatcher([
  (Pat(Const, name='x'), lambda x: f"{x.value}{dtype_suffix(x.dtype)}"),
  (Pat(Array, name='x'), lambda x: render_array(x.data, dtype_suffix(x.dtype))),
  (Pat(Var, name='x'), lambda ctx, x: f'{x.name}{ctx[x.name]}'),
  (Pat(Assign, name='x'), render_assign),
  (Pat(Store, name='x'), lambda ctx, x: f"{ctx[x.data]} = {ctx[x.value]};"),
  (Pat(Allocate, name='x'), lambda ctx, x: f"malloc({ctx[x.size]})"),
  (Pat(Free, name='x'), lambda ctx, x: f"free({ctx[x.var]});"),
  (Pat(Loop, name='x'), lambda ctx, x: f"for ({ctx[x.assign]} {ctx[x.idx]} < {ctx[x.stop]}; {ctx[x.idx]}++) {{\n{append_indent(ctx[x.scope])}\n}}"),
  (Pat(Where, name='x'), lambda ctx, x: f"({ctx[x.condition]}?{ctx[x.passed]}:{ctx[x.failed]})"),
  (Pat(Branch, name='x'), lambda ctx, x: f"if ({ctx[x.condition]}) {{\n{append_indent(ctx[x.passed])}\n}}" + ("" if x.failed == None else f"\nelse {{\n{append_indent(ctx[x.failed])}\n}}")),
  (Pat(Reshape, name='x'), lambda ctx, x: ctx[x.node]),
  (Pat(Load, name='x'), lambda ctx, x: f"*({ctx[x.data]} + {strip_parens(ctx[x.indexer]) if x.indexer._arg == '+' else ctx[x.indexer]})"),
  (Pat(Debug, sources=Pat(Var, name='x')),
   lambda ctx, x: f'puts(array_to_string({ctx[x]}, {x.dtype.itemsize}, (int[]){render_array(ctx[s] for s in x.shape)}, {len(x.shape)}, {x.dtype.fmt}_fmt));'),
  (Pat(Debug, sources=Pat(Node, name='x')), lambda ctx, x: f'printf("%{x.dtype.fmt}\\n", {ctx[x]});'),
  (Pat(BinaryOp, name='x'), lambda ctx, x: op_rendering[x.op](
    *[strip_parens(ctx[source]) if isinstance(source, BinaryOp) and source.op == x.op and x.op in NodeGroup.Associative else ctx[source] for source in x.sources]
  )),
])

def print_tracked_rewrites(tracker:list[tuple[Pat, Node, str]]):
  def get_rewrite(rewrite:tuple[Pat, Node, str], active:bool):
    ret = f"{rewrite[0]}\n{rewrite[2]}"
    if active: ret = "\x1b[1m" + "\n".join(f"> {line}" for line in ret.splitlines()) + "\x1b[0m"
    return ret
  navigate_history(lambda index: "\n\n".join(get_rewrite(rewrite, i==index) for i,rewrite in enumerate(tracker) if abs(i-index) <= 2), len(tracker))

def render(procedure:set[Node]):
  ctx:dict[Node, str] = {}
  functions = []
  non_terminal = set()
  terminal = {}
  tracker:list[tuple[Pat, Node, str]] = [] if TRACK_REWRITES else None
  for node in procedure:
    if isinstance(node, Var):
      ctx[node.name] = ctx.get(node.name, -1) + 1
    rewrite, pattern = render_patterns.rewrite(node, ctx)
    if TRACK_REWRITES: tracker.append((pattern, node, rewrite))
    if rewrite is None:
      print("RENDER: Failed to parse", node)
      continue
    ctx[node] = rewrite
    non_terminal.union(node.sources, node.shape)
    for n in node.sources + node.shape:
      if n in terminal: del terminal[n]
    if node not in non_terminal: terminal[node] = None
  
  if TRACK_REWRITES: print_tracked_rewrites(tracker)
  return '\n'.join(append_indent(ctx[x]) for x in terminal), '\n\n'.join(functions)