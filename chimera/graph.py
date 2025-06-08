from __future__ import annotations
from chimera.rewrite import PatternMatcher, Pat, RewriteContext, rewrite_graph, linearize
from chimera.helpers import DEBUG, prod
from chimera.nodes import *

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
  indices = [Var(0, f"idx") for _ in range(len(node.shape))]
  shape = node.shape
  node = Index(node, indices)
  for idx,dim in zip(reversed(indices), reversed(shape)):
    node = Loop(idx, dim, node)
  return node

def propagate_index(index:Index) -> Node:
  sources = map(lambda x: x if x.shape == () else Index(x, index.indices), index.data.sources)
  return index.data.copy(sources, index.data._arg, index.data.dtype, index.shape)

def merge_index(parent:Index, child:Index):
  merged_indices:list[Node] = []
  parent_index = 0
  for i in child.indices:
    assert isinstance(i, (Const, Slice, Var)), f"Invalid index type: {type(i).__name__}, value: {i}"
    if isinstance(i, Const) or parent_index == len(parent.indices):
      merged_indices.append(i)
      continue
    elif isinstance(i, Slice):
      merged_indices.append((parent.indices[parent_index] * i.step + i.begin))
    elif isinstance(i, Var):
      merged_indices.append(parent.indices[parent_index] * i)
    parent_index += 1
  for i in range(parent_index, min(len(parent.indices), len(child.indices) + len(child.shape))):
    merged_indices.append(parent.indices[i])
  return Index(child.data, merged_indices) if merged_indices else child.data

def lower_index_expand(index:Index, expand:Expand) -> Index:
  if expand.node.shape == (): return expand.node
  return Index(expand.node, tuple(0 if s == 1 else i for s, i in zip(expand.node.shape, index.indices[len(expand.shape)-len(expand.node.shape):])))

def lower_index_reshape(index:Index, reshape:Reshape) -> Index:
  offset = Const(0)
  for i, stride in zip(index.indices, reshape.strides): offset += i * stride
  new_indices = []
  offset = offset.simplify()
  for s in reversed(reshape.node.shape):
    new_indices.append(offset % s)
    offset = offset / s
  return Index(reshape.node, tuple(reversed(new_indices)))

def lower_index_permute(index:Index, permute:Permute):
  return Index(permute.node, tuple(index.indices[permute.inverse_permutation[i]] for i in range(len(index.indices))))

def lower_index_flip(index:Index, flip:Flip):
  return Index(flip.node, tuple((flip.shape[i] - Const(1)) - idx if i in flip.dims else idx for i,idx in enumerate(index.indices)))

base_rewrite = PatternMatcher([
  # Refactor print statements to use malloc/free
  (Pat(Debug, predicate=lambda x: x.data.shape != (), name="x"), refactor_debug),
  # Move array to assignments
  (Pat((BinaryOp, Expand, Reshape, Permute, Flip, Store, Index), name="parent"), assign_array),
])

index_collapse_rewrite = PatternMatcher([
    # Propagate indexing down the graph
    (Pat(Index, name="index", sources=Pat((BinaryOp, Store)), fuzzy_source_match=True), propagate_index),
    (Pat(Index, name="parent", sources=Pat(Index, name="child"), fuzzy_source_match=True), merge_index),
    (Pat(Index, name="index", sources=Pat(Expand, name="expand"), fuzzy_source_match=True), lower_index_expand),
    (Pat(Index, name="index", sources=Pat(Reshape, name="reshape"), fuzzy_source_match=True), lower_index_reshape),
    (Pat(Index, name="index", sources=Pat(Permute, name="permute"), fuzzy_source_match=True), lower_index_permute),
    (Pat(Index, name="index", sources=Pat(Flip, name="flip"), fuzzy_source_match=True), lower_index_flip),
    (Pat(Index, name="index", sources=Pat(Var, name="var"), fuzzy_source_match=True), lambda index, var: Load(var, index.indices)),
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