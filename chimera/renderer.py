from __future__ import annotations
from chimera.graph import Node, Print, BinaryOp, Index, Array, Var, Assign, Const, Loop, PatternMatcher, Pat
from chimera.dtype import dtypes

node_patterns: dict = {
  '+': lambda a,b: f"({a}+{b})",
  '-': lambda a,b: f"({a}-{b})",
  '*': lambda a,b: f"({a}*{b})",
  '/': lambda a,b: f"({a}/{b})",
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
def render_index(ctx, indexer:Index):
  rend = f"{ctx[indexer.data]}"
  # *(data0+idx0*stride0+idx1*stride1)
  print(indexer.indices)
  print(indexer.data.view.strides)
  for i,idx in enumerate(indexer.indices):
    rend = f"{rend} + {ctx[idx]}*{indexer.data.view.strides[i]}"
  return f"*({rend})"

def dtype_suffix(value, dtype) -> str: return f"{value}{'f' if dtype==float else ''}"
def append_indent(string:str, suffix:str='') -> str: return "\n".join(f"  {line}{suffix}" for line in string.splitlines())
def strip_parens(string:str) -> str: return string[1:-1]

# TODO fix ; placement in loops
render_patterns = PatternMatcher([
  (Pat(Const, name='x'), lambda x: dtype_suffix(x.value, x.dtype)),
  (Pat(Array, name='x'), lambda ctx, x: f"{{{','.join([dtype_suffix(v, x.dtype) for v in x.data])}}}"),
  (Pat(Var, name='x'), lambda ctx, x: f'{x.name}{ctx[x.name]}'),
  (Pat(Assign, name='x'), lambda ctx, x: f'{dtype_to_str[x.var.dtype]} {ctx[x.var]}{"[]" if x.var.shape != () else ""}={ctx[x.var.data]}'),
  (Pat(Loop, name='x'), lambda ctx, x: f"for ({ctx[x.assign]}; {ctx[x.idx]}<{ctx[x.stop]}; {ctx[x.idx]}+={ctx[x.step]}) {{\n {append_indent(ctx[x.scope], ';')}\n}}"),
  (Pat(Index, name='x'), lambda ctx, x: render_index(ctx,x)), # do strides in future and loop over all indices
  # (Pat(Call, name='x'), lambda ctx, x: f"{ctx[x.func]}({', '.join(ctx[arg] for arg in x.args)})"),
  (Pat(Print, sources=Pat(Node, name='x')), lambda ctx, x: r'printf("%d\n",' + f'{ctx[x]})'),
  (Pat(BinaryOp, name='x'), lambda ctx, x: node_patterns[x.op](
    *[strip_parens(ctx[source]) if isinstance(source, BinaryOp) and source.op == x.op and x.op in NodeGroup.Associative else ctx[source] for source in x.sources]
  )),
])

def render(procedure:set[Node]):
  ctx:dict[Node, str] = {}
  functions = []
  body = []
  for node in procedure:
    # if isinstance(node, Function):
    #   func_name = f"func{len(functions)}"
    #   rewrite = render_function(node, func_name, ctx)
    #   ctx[node] = func_name
    #   functions.append(rewrite)
    #   continue

    if isinstance(node, Var):
      ctx[node.name] = ctx.get(node.name, -1) + 1
    rewrite = render_patterns.rewrite(node, ctx=ctx)
    if rewrite is None:
      print("RENDER: Failed to parse", node)
      continue
    ctx[node] = rewrite
    if node.terminal:
      body.append(f'{append_indent(rewrite)};')
  return '\n'.join(body), '\n\n'.join(functions)