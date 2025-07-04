from __future__ import annotations
from typing import Callable, TypeVar
from chimera.helpers import DEBUG, TRACK_REWRITES, LOG_REWRITE_FAILURES, navigate_history
from chimera.nodes import *
import inspect, functools

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
  def __add__(self, x): return self._binop(Ops.ADD, x)
  def __sub__(self, x): return self._binop(Ops.SUB, x)
  def __mul__(self, x): return self._binop(Ops.MUL, x)
  def __mod__(self, x): return self._binop(Ops.MOD, x)
  def __truediv__(self, x): return self._binop(Ops.DIV, x)

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
      if p.match(node, vars):
        if LOG_REWRITE_FAILURES:
          try: rewritten = fxn(ctx=ctx, **vars) if has_ctx else fxn(**vars)
          except Exception as e:
            e.add_note(f"Pattern that caused exception:\n{p}\n\n{node.get_print_tree()}")
            raise e
        else: rewritten = fxn(ctx=ctx, **vars) if has_ctx else fxn(**vars)
        return rewritten, p
    return None, None

### Graph Rewrite ###

class RewriteContext:
  def __init__(self, track_rewrites=TRACK_REWRITES):
    self.variables:dict[Node, Var] = {} # Should not be here
    self.position:tuple[int] = ()
    self.tracker:TrackedRewrite|None = TrackedRewrite() if track_rewrites else None

class TrackedRewrite:
  def __init__(self):
    self._tracker:list[tuple[Pat, Node, tuple[int]]] = []
  def get_pattern(self, step:int): return self._tracker[step][0]
  def get_rewritten(self, step:int): return self._tracker[step][1]
  def get_position(self, step:int): return self._tracker[step][2]
  def track_step(self, pattern:Pat, rewritten:Node, position:tuple[int]):
    self._tracker.insert(len(self), (pattern, rewritten, position))
  def __len__(self): return len(self._tracker)

replacers:dict[PatternMatcher, dict[Node, Node]] = {}

def rewrite_tree(root:Node, rewriter:PatternMatcher, ctx:RewriteContext=None, track_rewrites:bool=TRACK_REWRITES):
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

  if ctx is None: ctx = RewriteContext(track_rewrites)

  if not (replace:=replacers.get(rewriter, {})): replacers[rewriter] = replace
  stack: list[tuple[Node, int, Node, tuple[int]]] = [(root, 0, root, ())]
  while stack:
    n, stage, new_n, position = stack.pop()
    if n in replace: continue
    if stage == 0:
      while new_n is not None:
        last_n, new_n, pattern = new_n, *rewriter.rewrite(new_n, ctx)
        if new_n == last_n or new_n == None: break
        new_n.print_tree()
        if ctx.tracker is not None:
          ctx.tracker.track_step(pattern, new_n, position)
      new_n = last_n
      stack.append((n, 1, new_n, position))
      for i, x in enumerate(reversed(new_n.sources)): stack.append((x, 0, x, position + (i,)))
    elif stage == 1:
      if (new_sources:=tuple([replace[x] for x in new_n.sources])) == new_n.sources:
        replace[n] = new_n
        continue
      new_sources_n = new_n.copy(new_sources)
      stack.append((n, 2, new_sources_n, position))
      stack.append((new_sources_n, 0, new_sources_n, position))
    else:
      replace[n] = replace[new_n]

  if track_rewrites: navigate_history(lambda i: _get_history_entry(ctx.tracker, root, i-1), len(ctx.tracker) + 1)
  return replace[root]

def print_procedure(nodes:list[Node]):
  for i,n in enumerate(nodes):
    formatted_parents = [nodes.index(x) if x in nodes else "--" for x in n.sources]
    print(f"{i:4d} {str(n):20s}: {str(n.dtype.name):8s} {str(formatted_parents):16s}")

def linearize(ast:Node) -> tuple[Node]:
  def _get_children_dfs(node:Node, visited:dict[Node, None]):
    if node in visited: return
    for dim in node.shape: _get_children_dfs(dim, visited)
    visited.update(dict.fromkeys(node.shape, None))

    if (scoped := isinstance(node, (Function, Range))):
      visited[ScopeBegin()] = None
      print(node)
    for source in node.sources:
      _get_children_dfs(source, visited)
      if isinstance(node, Block): visited[source] = None
    visited.update(dict.fromkeys(node.sources, None))

  visited:dict[Node, None] = {}
  _get_children_dfs(ast, visited)
  visited[ast] = None
  visited = tuple(visited)
  if DEBUG >= 2: print_procedure(visited)
  return visited