from __future__ import annotations
from chimera.patternmatcher import PatternMatcher, Pat, RewriteContext, rewrite_graph
from chimera.helpers import DEBUG, prod
from chimera.nodes import *

def print_procedure(nodes:list[Node]):
  for i,n in enumerate(nodes):
    formatted_parents = [nodes.index(x) if x in nodes else "--" for x in n.sources]
    print(f"{i:4d} {str(n):20s}: {str(n.dtype.name):8s} {str(formatted_parents):16s}")

def refactor_debug(ctx:RewriteContext, x:Debug) -> Var:
  var = Var(Reshape(Allocate(prod(x.data.shape, Const(1)), x.data.dtype), x.data.shape), "dbg")
  ctx.pre[x] = Assign(var)
  ctx.post[x] = Debug(var)
  ctx.post[x.data] = Free(var)
  return create_loop(Store(var, x.data))

def assign_array(ctx:RewriteContext, parent:Node) -> Node:
  sources = list(parent.sources)
  for i,arr in enumerate(sources): 
    if not isinstance(arr, Array): continue
    if arr not in ctx.pre:
      var = Var(arr, "arr")
      ctx.pre[arr] = Assign(var)
    sources[i] = ctx.pre[arr].var
  return None if sources == list(parent.sources) else parent.copy(sources)

def create_loop(node:Node) -> Loop:
  indices = [Var(0, f"idx{i}") for i in range(len(node.shape))]
  shape = node.shape
  node = Index(node, indices)
  for idx,dim in zip(reversed(indices), reversed(shape)):
    node = Loop(idx, dim, node)
  return node

def propagate_index(idx: Index) -> Node:
  sources = map(lambda x: x if x.shape == () else Index(x, idx.indices), idx.data.sources)
  return idx.data.copy(sources, idx.data._arg, idx.data.dtype, idx.shape)

def merge_index(idx1:Index, idx2:Index):
  merged_indices:list[Node] = []
  parent_index = 0
  for i in idx2.indices:
    assert isinstance(i, (Const, Slice, Var)), f"Invalid index type: {type(i).__name__}, value: {i}"
    if isinstance(i, Const) or parent_index == len(idx1.indices):
      merged_indices.append(i)
      continue
    elif isinstance(i, Slice):
      merged_indices.append((idx1.indices[parent_index] * i.step + i.begin))
    elif isinstance(i, Var):
      merged_indices.append(idx1.indices[parent_index] * i)
    parent_index += 1
  for i in range(parent_index, min(len(idx1.indices), len(idx2.indices) + len(idx2.shape))):
    merged_indices.append(idx1.indices[i])
  return Index(idx2.data, merged_indices) if merged_indices else idx2.data

def shrink(idx:Index, expand:Expand) -> Index:
  if expand.node.shape == (): return expand.node
  return Index(expand.node, tuple(0 if s == 1 else i for s, i in zip(expand.node.shape, idx.indices[len(expand.shape)-len(expand.node.shape):])))

def lower_index_reshape(idx:Index, reshape:Reshape) -> Index:
  offset = Const(0)
  for i, stride in zip(idx.indices, reshape.strides): offset += i * stride
  new_indices = []
  offset = offset.simplify()
  for s in reversed(reshape.node.shape):
    new_indices.append(offset % s)
    offset = offset / s
  return Index(reshape.node, tuple(reversed(new_indices)))

base_rewrite = PatternMatcher([
  # Refactor print statements to use malloc/free
  (Pat(Debug, predicate=lambda x: x.data.shape != (), name="x"), refactor_debug),
  # Move array to assignments
  (Pat((BinaryOp, Expand, Reshape, Store, Index), name="parent"), assign_array),
])

index_collapse_rewrite = PatternMatcher([
    # Propagate indexing down the graph
    (Pat(Index, name="idx", sources=Pat((BinaryOp, Store)), fuzzy_source_match=True), propagate_index),
    (Pat(Index, name="idx1", sources=Pat(Index, name="idx2"), fuzzy_source_match=True), merge_index),
    (Pat(Index, name="idx", sources=Pat(Expand, name="expand"), fuzzy_source_match=True), shrink),
    (Pat(Index, name="idx", sources=Pat(Reshape, name="reshape"), fuzzy_source_match=True), lower_index_reshape),
    (Pat(Index, name="idx", sources=Pat(Var, name="var"), fuzzy_source_match=True), lambda idx, var: Load(var, idx.indices)),
])

def apply_rewrite_passes(graph:Node) -> Node:
  # Base rewrite
  graph = rewrite_graph(graph, base_rewrite, (context:=RewriteContext()))
  graph = Program(tuple(context.pre.values()) + graph.sources + tuple(context.post.values()))

  # Index collapse rewrite
  graph = rewrite_graph(graph, index_collapse_rewrite)

  # Symbolic rewrite
  graph = graph.simplify()
  return graph

def linearize(ast:Node) -> tuple[Node]:
  def _get_children_dfs(node:Node, visited:dict[Node, None]):
    if node in visited: return
    for dim in node.shape:
      _get_children_dfs(dim, visited)
    for source in node.sources:
      _get_children_dfs(source, visited)
    visited.update(dict.fromkeys(node.shape, None))
    visited.update(dict.fromkeys(node.sources, None))

  visited:dict[Node, None] = {}
  _get_children_dfs(ast, visited)
  visited = tuple(visited)
  if DEBUG >= 2: print_procedure(visited)
  return visited

def parse_ast(ast:list[Node]|Node) -> tuple[Node]:
  if not isinstance(ast, Program): ast = Program(ast)
  if DEBUG:
    print("GRAPH")
    ast.print_tree()

  ast = apply_rewrite_passes(ast)

  if DEBUG >= 2:
    print("LOWERED GRAPH")
    ast.print_tree()
  return linearize(ast)