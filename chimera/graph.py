from __future__ import annotations
from typing import Callable, TypeVar
from chimera.helpers import DEBUG
from chimera.nodes import *
import inspect, functools

def print_graph(nodes:list[Node]):
  visited, var_count = set(), []
  for n in nodes: n.print_tree(visited, var_count)

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
    return f"Pat({self.type}, {self.name})"

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
  def rewrite(self, node:Node, ctx=None) -> any|None:
    for p,fxn,has_ctx in self.pdict.get(type(node), []):
      vars = {}
      if p.match(node, vars):
        return fxn(ctx=ctx, **vars) if has_ctx else fxn(**vars)
    return None

### Graph Rewrite ###

class RewriteContext:
  pre:dict[Node, Node] = {}
  post:dict[Node, Node] = {}

def refactor_print(ctx:RewriteContext, x:Print) -> Var:
  var = Var(Allocate(x.data), "prt")
  ctx.pre[x] = Assign(var)
  ctx.post[x] = Print(var)
  ctx.post[x.data] = Free(var)
  return create_loop(Store(var, x.data))
def assign_array(ctx:RewriteContext, parent:Node) -> Node:
  sources = list(parent.sources)
  for i,arr in enumerate(parent.sources): 
    if not isinstance(arr, Array): continue
    if arr not in ctx.pre:
      var = Var(arr, "arr")
      ctx.pre[arr] = Assign(var)
    sources[i] = ctx.pre[arr].var
  return None if sources == list(parent.sources) else parent.copy(sources)
def create_loop(node:Node) -> Loop:
  indices = [Var(0, "idx") for _ in range(len(node.shape))]
  shape = node.shape
  node = Index(node, indices)
  for idx,dim in zip(indices, shape):
    node = Loop(idx, dim, node)
  return node
def propagate_index(idx: Index) -> Node:
  sources = map(lambda x: x if x.shape == () else Index(x, idx.indexer), idx.data.sources)
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
  # Create ranges to lower indices in the graph
  # (Pat((BinaryOp, Store), predicate=lambda x: x.shape != (), name="x"), lambda x: create_loop(x)),
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

def rewrite_ast(ast:list[Node], rewriter:PatternMatcher, ctx=None) -> list[Node]:
  return [rewrite_graph(node, rewriter, ctx) for node in ast]
def rewrite_graph(n:Node, rewriter:PatternMatcher, ctx=None, replace:dict[Node, Node]=None) -> Node:
  if replace is None: replace = {}
  elif (rn := replace.get(n)) is not None: return rn
  new_n:Node = n; last_n:Node = None
  while new_n is not None:
    last_n, new_n = new_n, rewriter.rewrite(new_n, ctx)
  new_src = tuple(rewrite_graph(node, rewriter, ctx) for node in last_n.sources)
  replace[n] = ret = last_n if new_src == last_n.sources else rewrite_graph(last_n.copy(new_src), rewriter, ctx)
  return ret

def linearize(ast:list[Node]) -> set[Node]:
  def _get_children_dfs(node:Node, visited:dict[Node, None]):
    if node in visited: return
    for source in node.sources:
      _get_children_dfs(source, visited)
    visited.update(dict.fromkeys(node.sources, None))

  visited:dict[Node, None] = {}
  for node in ast:
    _get_children_dfs(node, visited)
    visited[node] = None
  visited = list(visited)
  if DEBUG >= 2: print_procedure(visited)
  return visited

def parse_ast(ast:list[Node]):
  if DEBUG:
    print("GRAPH")
    print_graph(ast)

  ast = rewrite_ast(ast, base_rewrite, (context:=RewriteContext()))
  ast = list(context.pre.values()) + ast + list(context.post.values())
  # ast = rewrite_ast(ast, symbolic)
  if DEBUG >= 2:
    print("LOWERED GRAPH")
    print_graph(ast)
  return linearize(ast)