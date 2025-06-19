from __future__ import annotations
from typing import Callable
from chimera.rewrite import PatternMatcher, Pat, RewriteContext, rewrite_tree, linearize
from chimera.helpers import DEBUG, prod
from chimera.nodes import *

def refactor_debug(x:Debug) -> Block:
  var = Var(Reshape(Allocate(prod(x.data.shape, Const(1)), x.data.dtype), x.data.shape), "dbg")
  return Block(Assign(var), create_loop_with_index(Store(var, x.data)), Debug(var), Free(var))

def lower_reduce(reduce:Reduce) -> Block:
  acc = Var(Reshape(Allocate(prod(reduce.shape, Const(1)), reduce.dtype), reduce.shape), "acc")
  return Block(Assign(acc), create_loop(lambda i: Store(acc, acc + Call(reduce.function, (acc, Index(reduce.value, i)))), reduce.shape))

def lower(x:Node):
  var = Var(Reshape(Allocate(prod(x.shape, Const(1)), x.dtype), x.shape), "temp")
  return Block(Assign(var), create_loop_with_index(Store(var, x)), var)

def assign_array(ctx:RewriteContext, parent:Node) -> Node:
  sources = list(parent.sources)
  assignments = []
  for i,arr in enumerate(sources): 
    if not isinstance(arr, Array): continue
    if arr in ctx.variables:
      sources[i] = ctx.variables[arr]
      continue
    ctx.variables[arr] = (var := Var(arr, "arr"))
    assignments.append(Assign(var))
    sources[i] = var

  if sources == list(parent.sources): return None
  return Block(*assignments, parent.copy(sources)) if assignments else parent.copy(sources)

def create_loop_with_index(node:Node):
  return create_loop(lambda indices: Index(node, indices), node.shape)
def create_loop(create_node:Callable[[tuple[Var]], Node], shape:tuple) -> Loop:
  indices = [Var(0, f"idx") for _ in range(len(shape))]
  node = create_node(indices)
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
  # Move array to assignments
  (Pat((BinaryOp, Expand, Reshape, Permute, Flip, Where, Store, Index, Debug), name="parent"), assign_array),
])

index_collapse_rewrite = PatternMatcher([
    # Propagate indexing down the graph
    (Pat(Index, name="index", sources=Pat((BinaryOp, Store, Where)), fuzzy_source_match=True), propagate_index),
    (Pat(Index, name="parent", sources=Pat(Index, name="child"), fuzzy_source_match=True), merge_index),
    (Pat(Index, name="index", sources=Pat(Expand, name="expand"), fuzzy_source_match=True), lower_index_expand),
    (Pat(Index, name="index", sources=Pat(Reshape, name="reshape"), fuzzy_source_match=True), lower_index_reshape),
    (Pat(Index, name="index", sources=Pat(Permute, name="permute"), fuzzy_source_match=True), lower_index_permute),
    (Pat(Index, name="index", sources=Pat(Flip, name="flip"), fuzzy_source_match=True), lower_index_flip),
    (Pat(Index, name="index", sources=Pat(Block, name="block"), fuzzy_source_match=True),
     lambda index, block: Block(*block.sources[:-1], Index(block.sources[-1], index.indices))),

    (Pat((Index, BinaryOp, Expand, Permute, Flip, Where), predicate=lambda x: x.shape != (), name="x"), lower),
    (Pat(Index, name="index", sources=Pat(Var, name="var"), fuzzy_source_match=True), lambda index, var: Load(var, index.indices)),
])

def apply_rewrite_passes(graph:Node) -> Node:
  # Base rewrite
  graph = rewrite_tree(graph, base_rewrite, RewriteContext())

  # Index collapse rewrite
  graph = rewrite_tree(graph, index_collapse_rewrite)

  # Symbolic rewrite
  graph = graph.simplify()
  return graph

def parse_ast(ast:list[Node]|Node) -> tuple[Node]:
  if isinstance(ast, list): ast = Block(*ast)
  if DEBUG:
    print("GRAPH")
    ast.print_tree()

  ast = apply_rewrite_passes(ast)

  if DEBUG >= 2:
    print("LOWERED GRAPH")
    ast.print_tree()
  return linearize(ast)