from __future__ import annotations
from chimera.graph import *
from chimera.dtype import dtypes

op_rendering: dict = {
  '+': lambda a,b: f"({a}+{b})",
  '-': lambda a,b: f"({a}-{b})",
  '*': lambda a,b: f"({a}*{b})",
  '/': lambda a,b: f"({a}/{b})",
  '%': lambda a,b: f"({a}%{b})",
}

class NodeGroup:
  Associative = {'+', '*'}
  Terminal = {Print, Loop}

class TypeGroup:
  Number = {dtypes.int32, dtypes.float32}

dtype_to_str = {dtypes.int32:'int', dtypes.float32:'float', None:'void'}

# def render_function(func:Function, name:str, ctx) -> tuple[str, str]:
#   arg_string = ', '.join(f'{dtype_to_str[arg.dtype]} {ctx[arg]}' for arg in func.args)
#   return f"{dtype_to_str[None]} {name}({arg_string}) {{\n{append_indent(ctx[func.body], ';')}\n}}"
def render_assign(ctx, x:Assign):
  # C initializes arrays and pointers with different syntax
  if isinstance(x.var.data, Array):
    return f'{dtype_to_str[x.var.dtype]} {ctx[x.var]}[] = {ctx[x.var.data]};'
  pointer = '*' if x.var.shape != () else ''
  return f'{dtype_to_str[x.var.dtype]}{pointer} {ctx[x.var]} = {ctx[x.var.data]};'
def render_array(data, suffix=''): return f"{{{','.join([f'{v}{suffix}' for v in data])}}}"
def dtype_suffix(dtype) -> str: return 'f' if dtype==dtypes.float32 else ''
def append_indent(string:str) -> str: return "\n".join(f"  {line}" for line in string.splitlines())
def strip_parens(string:str) -> str: return string[1:-1]

# TODO fix ; placement in loops
render_patterns = PatternMatcher([
  (Pat(Const, name='x'), lambda x: f"{x.value}{dtype_suffix(x.dtype)}"),
  (Pat(Array, name='x'), lambda x: render_array(x.data, dtype_suffix(x.dtype))),
  (Pat(Var, name='x'), lambda ctx, x: f'{x.name}{ctx[x.name]}'),
  (Pat(Assign, name='x'), render_assign),
  (Pat(Store, name='x'), lambda ctx, x: f"{ctx[x.data]} = {ctx[x.value]};"),
  (Pat(Allocate, name='x'), lambda x: f"malloc({x.size})"),
  (Pat(Free, name='x'), lambda ctx, x: f"free({ctx[x.var]});"),
  (Pat(Loop, name='x'), lambda ctx, x: f"for ({ctx[x.assign]} {ctx[x.idx]} < {ctx[x.stop]}; {ctx[x.idx]}++) {{\n {append_indent(ctx[x.scope])}\n}}"),
  (Pat((Expand, Reshape), name='x'), lambda ctx, x: ctx[x.node]),
  (Pat(Index, name='x'), lambda ctx, x: f"*({ctx[x.data]} + {strip_parens(ctx[x.indexer]) if x.indexer._arg == "+" else ctx[x.indexer]})"),
  # (Pat(Call, name='x'), lambda ctx, x: f"{ctx[x.func]}({', '.join(ctx[arg] for arg in x.args)})"),
  (Pat(Print, sources=Pat(Node, name='x')), lambda ctx, x: f'puts(array_to_string({ctx[x]}, {x.dtype.itemsize}, {x.view.size}, (int[]){render_array(x.shape)}, {len(x.shape)}, (int[]){render_array(x.view.strides)}, "%{x.dtype.fmt}", {x.dtype.fmt}_fmt));'),
  (Pat(BinaryOp, name='x'), lambda ctx, x: op_rendering[x.op](
    *[strip_parens(ctx[source]) if isinstance(source, BinaryOp) and source.op == x.op and x.op in NodeGroup.Associative else ctx[source] for source in x.sources]
  )),
])

def render(procedure:set[Node]):
  ctx:dict[Node, str] = {}
  functions = []
  non_terminal = set()
  terminal = {}
  for node in procedure:
    if isinstance(node, Var):
      ctx[node.name] = ctx.get(node.name, -1) + 1
    rewrite = render_patterns.rewrite(node, ctx)
    if rewrite is None:
      print("RENDER: Failed to parse", node)
      continue
    ctx[node] = rewrite
    non_terminal.union(node.sources)
    for source in node.sources:
      if source in terminal: del terminal[source]
    if node not in non_terminal: terminal[node] = None

  return '\n'.join(append_indent(ctx[x]) for x in terminal), '\n\n'.join(functions)