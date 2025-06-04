from __future__ import annotations
from typing import Callable, TypeVar
from chimera.helpers import DEBUG, TRACK_REWRITES, navigate_history, canonicalize_strides
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
  def rewrite(self, node:Node, ctx=None) -> tuple[any|None, Pat|None]:
    for p,fxn,has_ctx in self.pdict.get(type(node), []):
      vars = {}
      if p.match(node,  vars):
        rewritten = fxn(ctx=ctx, **vars) if has_ctx else fxn(**vars)
        return rewritten, p
    return None, None

### Graph Rewrite ###

class RewriteContext:
  def __init__(self):
    self.pre:dict[Node, Node] = {}
    self.post:dict[Node, Node] = {}
    self.position:tuple[int] = ()
    self.tracker:TrackedRewrite|None = TrackedRewrite() if TRACK_REWRITES else None

class TrackedRewrite:
  def __init__(self):
    self._tracker:list[tuple[Pat, Node, tuple[int]]] = []
  def get_pattern(self, step:int): return self._tracker[step][0]
  def get_rewritten(self, step:int): return self._tracker[step][1]
  def get_position(self, step:int): return self._tracker[step][2]
  def track_step(self, pattern:Pat, rewritten:Node, position:tuple[int], step:int=None):
    self._tracker.insert(len(self) if step is None else step, (pattern, rewritten, position))
  def __len__(self): return len(self._tracker)

def refactor_debug(ctx:RewriteContext, x:Debug) -> Var:
  var = Var(Allocate(x.data.shape, x.data.dtype), "dbg")
  ctx.pre[x] = Assign(var)
  ctx.post[x] = Debug(var)
  ctx.post[x.data] = Free(var)
  return create_loop(ctx, Store(var, x.data))
def assign_array(ctx:RewriteContext, parent:Node) -> Node:
  sources = list(parent.sources)
  for i,arr in enumerate(sources): 
    if not isinstance(arr, Array): continue
    if arr not in ctx.pre:
      var = Var(arr, "arr")
      ctx.pre[arr] = Assign(var)
    sources[i] = ctx.pre[arr].var
  return None if sources == list(parent.sources) else parent.copy(sources)
def create_loop(ctx:RewriteContext, node:Node) -> Loop:
  print(node.shape)
  indices = [Var(0, "idx") for _ in range(len(node.shape))]
  shape = node.shape
  node = Index(node, indices)
  for idx,dim in zip(indices, reversed(shape)):
    node = Loop(idx, dim, node)
  return node
def propagate_index(idx: Index) -> Node:
  sources = map(lambda x: x if x.shape == () else Index(x, idx.indices), idx.data.sources)
  return idx.data.copy(sources, idx.data._arg, idx.data.dtype, idx.view)
def shrink(idx:Index, expand:Expand) -> Index:
  return Index(expand.node, canonicalize_strides(expand.node.shape, idx.indices))
def lower_index_reshape(idx:Index, reshape:Reshape) -> Index:
  offset = Const(0)
  for i, stride in zip(idx.indices[::-1], strides_for_shape(reshape.shape)): offset += i * stride
  new_indices = []
  for s in reversed(reshape.node.shape):
    new_indices.append(offset % s)
    offset = offset / s
  return Index(reshape.node, tuple(new_indices))

base_rewrite = PatternMatcher([
  # Refactor print statements to use malloc/free
  (Pat(Debug, predicate=lambda x: x.data.shape != (), name="x"), refactor_debug),
  # Move array to assignments
  (Pat((BinaryOp, Expand, Reshape, Store, Index), name="parent"), assign_array),
])

index_collapse_rewrite = PatternMatcher([
    # Propagate indexing down the graph
    (Pat(Index, name="idx", sources=Pat((BinaryOp, Store)), fuzzy_source_match=True), propagate_index),
    (Pat(Index, name="idx1", sources=Pat(Index, name="idx2"), fuzzy_source_match=True),
     lambda idx1, idx2: Index(idx2.data, idx2.indices + idx1.indices[len(idx2.shape):])),
    (Pat(Index, name="idx", sources=Pat(Expand, name="expand"), fuzzy_source_match=True), shrink),
    (Pat(Index, name="idx", sources=Pat(Reshape, name="reshape"), fuzzy_source_match=True), lower_index_reshape),
    (Pat(Index, name="idx", sources=Pat(Var, name="var"), fuzzy_source_match=True), lambda idx, var: Load(var, idx.indices)),
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

def recurse_rewrite_graph(n:Node, rewriter:PatternMatcher, ctx:RewriteContext, position:tuple[int]=None, replace:dict[Node, Node]=None) -> Node:
  assert isinstance(n, Node), f"Expected Node, got {n}"
  if replace is None: replace = {}
  elif (rn := replace.get(n)) is not None:
    return rn
  if position is None: position = ctx.position
  else: ctx.position = position
  new_n:Node = n; last_n:Node = None
  while new_n is not None:
    if ctx.tracker is not None: tracker_step = len(ctx.tracker)
    last_n, new_n, pattern = new_n, *rewriter.rewrite(new_n, ctx)
    if new_n == last_n: break
    if ctx.tracker is not None and new_n is not None:
      ctx.tracker.track_step(pattern, new_n, position, tracker_step)
  assert isinstance(last_n, Node), f"Rewriter has to return a {Node.__name__} but returned {last_n} from {n}.\nLast pattern: {pattern}"
  new_src = tuple(recurse_rewrite_graph(node, rewriter, ctx, position + (i,), replace) for i,node in enumerate(last_n.sources))
  replace[n] = ret = last_n if new_src == last_n.sources else recurse_rewrite_graph(last_n.copy(new_src), rewriter, ctx, position, replace)
  return ret

def rewrite_graph(graph:Node, rewriter:PatternMatcher, ctx:RewriteContext=None) -> Node:  
  @functools.cache
  def _get_graph(tracker:TrackedRewrite, root:Node, i:int) -> Node:
    if i <= -1: return root
    node = _get_graph(tracker, root, i-1)
    node_stack = [node]
    for pos in tracker.get_position(i)[:-1]:
      node = node.sources[pos]
      node_stack.append(node)
    new_graph = tracker.get_rewritten(i)
    for pos,n in zip(tracker.get_position(i)[::-1], node_stack[::-1]):
      new_graph = n.copy(n.sources[:pos] + (new_graph,) + n.sources[pos+1:])
    return new_graph
  @functools.cache
  def _get_history_entry(tracker:TrackedRewrite, root:Node, i:int) -> str:
    def format(node:Node, previous:set[Node]) -> str:
      return node.__repr__() if node in previous else f"\x1b[1m{node}\x1b[0m"
    pattern = f"{tracker.get_pattern(i)}\n" if i >= 0 else ""
    return f"{pattern}{_get_graph(tracker, root, i).get_print_tree(set(), [], lambda x: format(x, linearize(_get_graph(tracker, root, i-1))))}"

  if ctx is None: ctx = RewriteContext()
  rewrite = recurse_rewrite_graph(graph, rewriter, ctx)
  if TRACK_REWRITES: navigate_history(lambda i: _get_history_entry(ctx.tracker, graph, i-1), len(ctx.tracker) + 1)
  return rewrite

def apply_rewrite_passes(graph:Node) -> Node:
  # Base rewrite
  graph = rewrite_graph(graph, base_rewrite, (context:=RewriteContext()))
  graph = Program(tuple(context.pre.values()) + graph.sources + tuple(context.post.values()))

  # Index collapse rewrite
  graph = rewrite_graph(graph, index_collapse_rewrite)

  # Symbolic rewrite
  graph = rewrite_graph(graph, symbolic)
  return graph

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

  ast = apply_rewrite_passes(ast)

  if DEBUG >= 2:
    print("LOWERED GRAPH")
    ast.print_tree()
  return linearize(ast)