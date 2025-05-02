from __future__ import annotations
from typing import Callable, TypeVar
from chimera.helpers import DEBUG, TRACK_REWRITES, navigate_history
from chimera.nodes import *
import inspect, functools

def print_procedure(nodes:list[Node]):
  for i,n in enumerate(nodes):
    formatted_parents = [nodes.index(x) if x in nodes else "--" for x in n.sources]
    print(f"{i:4d} {str(n):20s}: {str(n.dtype.name):8s} {str(formatted_parents):16s}")

### Pattern Matching ###

T = TypeVar('T', bound=Node)

class Pat():
  __slots__ = ["type", "dtype", "predicate", "name", "sources", "fuzzy_source_match"]
  def __init__(self, type:type[T]|tuple[type[T]]=None,
               dtype:DType|tuple[DType]=None, predicate:Callable[[T], bool]=None,
               name:str=None, sources:Pat|tuple[Pat]=None, fuzzy_source_match:bool=False):
    if type is None: type = (Node,)
    self.type:tuple[type] = type if isinstance(type, tuple) else (type,)
    self.dtype:tuple[DType] = dtype if isinstance(dtype, tuple) or dtype is None else (dtype,)
    self.predicate = predicate
    self.name = name
    self.sources:tuple[Pat] = sources if isinstance(sources, tuple) or sources is None else (sources,)
    self.fuzzy_source_match = fuzzy_source_match

  def match(self, node:Node, args:dict[str, Node|tuple[Node]]) -> bool:
    if (not any(isinstance(node, t) for t in self.type)) or \
       (self.name is not None and args.setdefault(self.name, node) is not node) or \
       (self.dtype is not None and node.dtype not in self.dtype) or \
       (self.predicate is not None and not self.predicate(node)): return False
    if self.sources is not None:
      if self.fuzzy_source_match:
        if len(self.sources) > len(node.sources): return False
        # Only checks the first sources. Should maybe check all of them?
        if not all(pat.match(source, args) for pat, source in zip(self.sources, node.sources[:len(self.sources)])): return False
      else:
        if len(self.sources) != len(node.sources): return False
        if not all(pat.match(source, args) for pat, source in zip(self.sources, node.sources)): return False
    return True
  def __repr__(self):
    types = "" if self.type is None else f"({','.join(t.__name__ for t in self.type)})" if len(self.type) > 1 else self.type[0].__name__
    data_types = "" if self.dtype is None else ", " + (f"({','.join(t.__name__ for t in self.dtype)})" if len(self.dtype) > 1 else self.dtype[0].__name__)
    name = "" if self.name is None else f", '{self.name}'"
    sources = "" if self.sources is None else ''.join(f"\n  {s}" for s in self.sources)
    return f"Pat({types}{data_types}{name}){sources}"

  def _binop(self, op:str, x): return Pat(BinaryOp, predicate=lambda x: x.op==op, sources=(self, Pat.to_pat(x)))
  def __add__(self, x): return self._binop('+', x)
  def __sub__(self, x): return self._binop('-', x)
  def __mul__(self, x): return self._binop('*', x)
  def __mod__(self, x): return self._binop('%', x)
  def __truediv__(self, x): return self._binop('/', x)

  @staticmethod
  @functools.cache
  def var(name:str=None, dtype:DType|tuple[DType]=None): return Pat(dtype=dtype, name=name)
  @staticmethod
  @functools.cache
  def cvar(name:str=None, dtype:DType|tuple[DType]=None): return Pat(Const, dtype=dtype, name=name)
  @staticmethod
  def const(value, dtype:DType|tuple[DType]=None): return Pat(Const, dtype=dtype, predicate=lambda x: x.value==value)
  @staticmethod
  def to_pat(x): return x if isinstance(x, Pat) else Pat.const(x)

class PatternMatcher:
  def __init__(self, patterns:list[tuple[Pat, Callable[[any, Node], any]]]):
    self.patterns = patterns
    self.pdict: dict[Node, list[tuple[Pat, Callable[[any, Node], any], bool]]] = {}
    for pat, fxn in self.patterns:
      for node_type in pat.type: self.pdict.setdefault(node_type, []).append((pat, fxn, 'ctx' in inspect.signature(fxn).parameters))
  def rewrite(self, node:Node, ctx=None, tracker:list[tuple[Pat, Node, any]]=None) -> any|None:
    for p,fxn,has_ctx in self.pdict.get(type(node), []):
      vars = {}
      if p.match(node,  vars):
        rewritten = fxn(ctx=ctx, **vars) if has_ctx else fxn(**vars)
        if tracker is not None and rewritten is not None: tracker.append((p, node, rewritten))
        return rewritten 
    return None

### Graph Rewrite ###

class RewriteContext:
  def __init__(self):
    self.pre:dict[Node, Node] = {}
    self.post:dict[Node, Node] = {}

def refactor_print(ctx:RewriteContext, x:Print) -> Var:
  var = Var(Allocate(x.data), "_print")
  ctx.pre[x] = Assign(var)
  ctx.post[x] = Print(var)
  ctx.post[x.data] = Free(var)
  return create_loop(Store(var, x.data))
def assign_array(ctx:RewriteContext, parent:Node) -> Node:
  sources = list(parent.sources)
  for i,arr in enumerate(sources): 
    if not isinstance(arr, Array): continue
    if arr not in ctx.pre:
      var = Var(arr, "_arr")
      ctx.pre[arr] = Assign(var)
    sources[i] = ctx.pre[arr].var
  return None if sources == list(parent.sources) else parent.copy(sources)
def create_loop(node:Node) -> Loop:
  indices = [Var(0, "_idx") for _ in range(len(node.shape))]
  shape = node.shape
  node = Index(node, indices)
  for idx,dim in zip(indices, shape):
    node = Loop(idx, dim, node)
  return node
def propagate_index(idx: Index) -> Node:
  sources = map(lambda x: x if x.shape == () else idx.copy((x, idx.indexer)), idx.data.sources)
  return idx.data.copy(sources, idx.data._arg, idx.data.dtype, idx.view)
def rewrite_index(idx:Index, dim_node:Reshape|Expand) -> Index:
  new_indices = []
  remaining = idx.indexer
  for stride in dim_node.node.view.strides:
    new_indices.append(remaining / stride)
    remaining = remaining % stride  
  return Index(dim_node.node, new_indices)

base_rewrite = PatternMatcher([
  # Refactor print statements to use malloc/free
  (Pat(Print, predicate=lambda x: x.data.shape != (), name="x"), refactor_print),
  # Propagate indexing down the graph
  (Pat(Index, name="idx", sources=(Pat(Reshape, name="dim_node")), fuzzy_source_match=True), rewrite_index),
  (Pat(Index, name="idx", sources=(Pat((BinaryOp, Store))), fuzzy_source_match=True), propagate_index),
  (Pat(Index, name="idx1", sources=(Pat(Index, name="idx2")), fuzzy_source_match=True), lambda idx1,idx2: Index(idx2.data, idx1.indices + idx2.indices)),
  # Move array to assignments
  (Pat((Index, BinaryOp, Expand, Reshape), name="parent"), assign_array),
])

symbolic = PatternMatcher([
  # Constant folding
  (Pat.cvar("x1") + Pat.cvar("x2"), lambda x1,x2: Const(x1.value + x2.value)),
  (Pat.cvar("x1") * Pat.cvar("x2"), lambda x1,x2: Const(x1.value * x2.value)),

  # Move constants to the end
  (Pat.cvar("x1") + Pat.var("x2"), lambda x1,x2: x2 + x1),
  (Pat.cvar("x1") * Pat.var("x2"), lambda x1,x2: x2 * x1),

  (Pat(Loop, predicate=lambda x: isinstance(x.idx.data, Const) and x.stop.value - x.idx.data.value == 1, name="x"),
   lambda x: x.scope),

  (Pat.var("x") + 0, lambda x: x), # x+0 -> x
  (Pat.var("x") * 1, lambda x: x), # x*1 -> x
  (Pat() * 0, lambda: Const(0)), # x*0 -> 0
  (Pat.var("x") / Pat.var("x"), lambda: Const(1)), # x/x -> 1

  (Pat.var("y") / Pat.var("x") * Pat.var("x"), lambda y, x: y), # (y/x)*x -> y
  (Pat.var("y") * Pat.var("x") / Pat.var("x"), lambda y, x: y), # (y*x)/x -> y
])

def rewrite_ast(ast:Node, rewriter:PatternMatcher, ctx=None) -> Node:
  def rewrite_graph(n:Node, rewriter:PatternMatcher, ctx, tracker:list, position:tuple[int], replace:dict[Node, Node]=None) -> Node:
    if replace is None: replace = {}
    elif (rn := replace.get(n)) is not None:
      return rn
    new_n:Node = n; last_n:Node = None
    while new_n is not None:
      last_n, new_n = new_n, rewriter.rewrite(new_n, ctx, tracker)
      if tracker is not None and new_n is not None: tracker[-1] = tracker[-1] + (position,)
    new_src = tuple(rewrite_graph(node, rewriter, ctx, tracker, position + (i,), replace) for i,node in enumerate(last_n.sources))
    replace[n] = ret = last_n if new_src == last_n.sources else rewrite_graph(last_n.copy(new_src), rewriter, ctx, tracker, position, replace)
    return ret
  
  @functools.cache
  def _get_graph(tracker:tuple, root:Node, i:int) -> Node:
    if i <= -1: return root
    node = _get_graph(tracker, root, i-1)
    node_stack = [node]
    for pos in tracker[i][3][:-1]:
      node = node.sources[pos]
      node_stack.append(node)
    new_graph = tracker[i][2]
    for pos,n in zip(tracker[i][3][::-1], node_stack[::-1]):
      new_graph = n.copy(n.sources[:pos] + (new_graph,) + n.sources[pos+1:])
    return new_graph
  @functools.cache
  def _get_history_entry(tracker:tuple, root:Node, i:int) -> str:
    def format(node:Node, previous:set[Node]) -> str:
      return node.__repr__() if node in previous else f"\x1b[1m{node.__repr__()}\x1b[0m"
    pattern = f"{tracker[i][0]}\n" if i >= 0 else ""
    return f"{pattern}{_get_graph(tracker, root, i).get_print_tree(set(), [], lambda x: format(x, linearize(_get_graph(tracker, root, i-1))))}"

  tracker:list[tuple[Pat, Node, Node, tuple[int]]] = [] if TRACK_REWRITES else None
  rewrite = rewrite_graph(ast, rewriter, ctx, tracker, ())
  if TRACK_REWRITES:
    tracker = tuple(tracker)
    navigate_history(lambda i: _get_history_entry(tracker, ast, i-1), len(tracker) + 1)
  return rewrite

def linearize(ast:Node) -> tuple[Node]:
  def _get_children_dfs(node:Node, visited:dict[Node, None]):
    if node in visited: return
    for source in node.sources:
      _get_children_dfs(source, visited)
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

  ast = rewrite_ast(ast, base_rewrite, (context:=RewriteContext()))
  ast = Program(tuple(context.pre.values()) + ast.sources + tuple(context.post.values()))
  ast = rewrite_ast(ast, symbolic)
  if DEBUG >= 2:
    print("LOWERED GRAPH")
    ast.print_tree()
  return linearize(ast)