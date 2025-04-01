from __future__ import annotations
from typing import Callable
from chimera.dtype import DType, dtypes
from chimera.helpers import DEBUG, fully_flatten, get_shape, all_same, all_instance, listed
from chimera.view import View
import inspect

def print_graph(nodes:list[Node]):
  var_count = []
  for n in nodes: n.print_tree(var_count)

def print_procedure(nodes:list[Node]):
  for i,n in enumerate(nodes):
    formatted_parents = [nodes.index(x) if x in nodes else "--" for x in n.sources]
    print(f"{i:4d} {str(n):20s}: {str(n.dtype):8s} {str(formatted_parents):16s}")


### Graph Nodes ###

class Node():
  terminal:bool = False
  _sources:list[Node] = []
  _dtype:DType = dtypes.void
  _view:View = View.create()
  @property
  def sources(self) -> tuple[Node]: return self._sources
  @property
  def dtype(self) -> DType: return self._dtype
  @property
  def view(self) -> View: return self._view
  @property
  def shape(self) -> tuple[int, ...]: return self._view.shape

  def update_sources(self, sources):
    sources = listed(sources)
    assert not self.sources or len(self.sources) == len(sources), f"New sources must be of same length as original: Expected {len(self._sources)}, Actual {len(sources)}"
    self._sources = sources

  def print_tree(self, var_count:list[Node]=None):
    if not var_count: var_count = []
    string = self.__repr__() + "\n"
    for i, source in enumerate(self.sources): string += source._get_print_tree(var_count, i == len(self.sources) - 1)
    print(string)

  def _get_print_tree(self, var_count:list[Node], is_last:bool, indent:str = ''):
    node_str = self.__repr__()
    if isinstance(self, Var):
      if self not in var_count: var_count.append(self)
      node_str += f" {var_count.index(self)}"
    tree_str = indent + ("└─" if is_last else "├─") + node_str + "\n"
    indent += "  " if is_last else "│ "
    for i, source in enumerate(self.sources): tree_str += source._get_print_tree(var_count, i == len(self.sources) - 1, indent)
    return tree_str

  def __repr__(self): return self.__class__.__name__

class Const(Node):
  def __init__(self, value):
    assert isinstance(value, (int, float, Const)), f"Const node can only have values of type int, float, Const. Type was {type(value)}"
    self._value = value.value if isinstance(value, Const) else value
    self._dtype = dtypes.python_to_dtype[type(self.value)]
  @property
  def value(self): return self._value
  def __repr__(self): return f"Const {self.value}"

class Var(Node):
  def __init__(self, data:Node, name:str="var"):
    self.update_sources(data if isinstance(data, Node) else Const(data))
    self._name = name
  @property
  def name(self) -> str: return self._name
  @property
  def data(self) -> Node: return self.sources[0]
  def __repr__(self): return self.name
  def update_sources(self, sources):
    super().update_sources(sources)
    assert self.data, "self.data should not be none"
    self._view = self.data.view
    self._dtype = self.data.dtype

class Assign(Node):
  def __init__(self, var:Var):
    self._sources = [var]
  @property
  def var(self): return self.sources[0]

# class SymbolicIndex(Node):
#   def __init__(self, name: str):
#     self.name = name
#   def __repr__(self): return f"Idx {self.name}"

class Array(Node):
  def __init__(self, data:list):
    self._view = View.create(get_shape(data))
    self._data = fully_flatten(data)
    assert all_same([type(d) for d in data]), f"Array must contain only one type but got {data}"
    self._dtype = dtypes.python_to_dtype[type(self.data[0])]
  @property
  def data(self) -> Node: return self._data

class Index(Node):
  def __init__(self, data:Node, indices):
    self.update_sources([data] + [i if isinstance(i, Node) else Const(i) for i in listed(indices)])
  @property
  def data(self) -> Node: return self.sources[0]
  @property
  def indices(self) -> Node: return self.sources[1:]
  def update_sources(self, sources):
    super().update_sources(sources)
    assert len(self.indices) <= len(self.data.shape), f"Too many indices {self.indices} for {self.data.shape}"
    self._view = View.create(self.data.shape[len(self.indices):]) #TODO probs not correct
    self._dtype = self.data.dtype

class Loop(Node):
  def __init__(self, start:Var|Const, stop:Const, step:Const, scope:Node):
    if not isinstance(start, Var): start = Var(Const(start), "idx")
    stop, step = Const(stop), Const(step)
    self.update_sources([Assign(start), stop, step, scope])
  @property
  def assign(self) -> Assign: return self.sources[0]
  @property
  def idx(self) -> Var: return self.assign.var
  @property
  def stop(self) -> Const: return self.sources[1]
  @property
  def step(self) -> Const: return self.sources[2]
  @property
  def scope(self) -> Node: return self.sources[3]
  def update_sources(self, sources):
    super().update_sources(sources)
    assert isinstance(self.assign, Assign) and all_instance((self.stop, self.step), Const), f"Sources are of wrong types {[type(x) for x in sources]}"
    self._view = self.scope.view

class BinaryOp(Node):
  def __init__(self, op, left:Node, right:Node):
    self.update_sources([left, right])
    self._op = op
  @property
  def op(self): return self._op
  @property
  def left(self): return self.sources[0]
  @property
  def right(self): return self.sources[1]
  def update_sources(self, sources):
    super().update_sources(sources)
    shape = self.left.shape
    # Need to broadcast in the future
    if len(self.left.shape) < len(self.right.shape): shape = self.right.shape
    elif len(self.left.shape) == len(self.right.shape): tuple(max(l, r) for l, r in zip(self.left.shape, self.right.shape))
    self._view = View.create(shape)
    self._dtype = self.left.dtype
  def __repr__(self): return f"{super().__repr__()} {self.op}"

# class Function(Node):
#   def __init__(self, args, body):
#     self.args = (args,) if isinstance(args, Node) else args
#     self.body = body
#   @property
#   def sources(self): return self.args + (self.body,)
#   @sources.setter
#   def sources(self, sources): self.args, self.body = sources
#   @property
#   def dtype(self): return self.body.dtype
#   @property
#   def dtype(self): return self.body.shape

# class Call(Node):
#   def __init__(self, func:Function, args:tuple[Node]):
#     self.func = func
#     self.args = (args,) if isinstance(args, Node) else args
#   @property
#   def sources(self): return (self.func) + self.args
#   @sources.setter
#   def sources(self, sources): self.func, self.args = sources
#   @property
#   def dtype(self): return self.func.dtype
#   @property
#   def shape(self): return self.func.shape

class Print(Node):
  def __init__(self, data):
    self.update_sources(data)
  @property
  def data(self): return self.sources[0]
  def update_sources(self, sources):
    super().update_sources(sources)
    self._view = self.data.view

### Pattern Matching ###

class Pat():
  __slots__ = ["type", "predicate",  "name", "sources"]
  def __init__(self, type:type|tuple[type], predicate:Callable[[Node], bool]=None, name:str=None, sources:Pat|tuple[Pat]=None):
    self.type:tuple[type] = type if isinstance(type, tuple) or type is None else (type,)
    self.predicate = predicate
    self.name = name
    self.sources:tuple[Pat] = sources if isinstance(sources, tuple) or sources is None else (sources,)
  def match(self, node:Node, args:dict[str, Node|tuple[Node]]) -> bool:
    if (self.type is not None and not any(isinstance(node, t) for t in self.type)) or \
     (self.predicate is not None and not self.predicate(node)): return False
    if self.sources is not None:
      if len(self.sources) != len(node.sources): return False
      if not all([pat.match(source, args) for pat, source in zip(self.sources, node.sources)]): return False
    if self.name is not None: args[self.name] = node
    return True
  def __repr__(self):
    return f"Pat({self.type}, {self.name})"

class PatternMatcher:
  def __init__(self, patterns:list[tuple[Pat, callable]]):
    self.patterns = patterns
    self.pdict: dict[Node, list[tuple[Pat, Callable[[Node], str], bool]]] = {}
    for pat, fxn in self.patterns:
      for node_type in pat.type: self.pdict.setdefault(node_type, []).append((pat, fxn, 'ctx' in inspect.signature(fxn).parameters))
  def rewrite(self, node:Node, ctx=None) -> any|None:
    for p,fxn,has_ctx in self.pdict.get(type(node), []):
      vars = {}
      if p.match(node, vars):
        return fxn(ctx=ctx, **vars) if has_ctx else fxn(**vars)
    return None

# base_graph_rewrite = PatternMatcher([

# ])

# def rewrite_graph(ast:list[Node], rewriter:PatternMatcher):
#   i = 0
#   while i < len(ast):
#     rewrite:Node = rewriter.rewrite(ast[i])
#     if rewrite:
#       ast[i] = rewrite
#       continue
#     rewrite_graph(ast[i].sources, rewriter)
#     i += 1

### Graph Rewrite ###

def rewrite_arrays(graph:list[Node], assignments:dict[Array, Assign]=None) -> list[Node]:
  if (first_call := assignments is None): assignments = {}
  for node in graph:
    if isinstance(node, (Assign, Var)): continue
    new_sources = list(node.sources)
    for i, source in enumerate(node.sources):
      if isinstance(source, Array):
        if source in assignments:
          var = assignments[source].var
        else:
          var = Var(source, "arr")
          assign = Assign(var)
          assignments[source] = assign
        source = var
        new_sources[i] = var
    node.update_sources(new_sources)
    rewrite_arrays(node.sources, assignments)
  if first_call: return list(assignments.values()) + graph

# Propagates an Index node down the graph if its children are not data nodes (Array / Var)
# and combines indexing nodes into a single Index node
def propagate_indexing(index:Index, current:Node=None) -> Node:
  if current is None: current = index
  if isinstance(current, Const): return current
  if isinstance(current, Index):
    if index is not current:
      unaccounted_indices = len(current.shape) - len(current.indices)
      if unaccounted_indices <= 0: index = current
      else: index = Index(current.data, current.indices + index.indices[:unaccounted_indices])
    # Early exit if current is a data node
    if isinstance(index.data, (Array, Var)): return index
    current = propagate_indexing(index, index.data)
  if isinstance(current, (Array, Var)): return Index(current, index.indices)
      
  current.update_sources(propagate_indexing(index, s) for s in current.sources)
  return current

# Converts Array into Index nodes and adds Range nodes
def lower_graph(graph:list[Node], indices:list[Assign]=[]) -> list[Node]:
  lowered_graph = []
  for node in graph:
    if isinstance(node, Index):
      node = propagate_indexing(node)
    if isinstance(node, (Const, Index, Assign)):
      lowered_graph.append(node)
      continue
    new_indices = [Var(0, "idx") for _ in range(len(node.shape) - len(indices))]
    dims = node.shape
    if isinstance(node, (Array, Var)):
      node = Index(node, indices + new_indices)
    elif node.sources:
      node.update_sources(lower_graph(node.sources, indices + new_indices))
    for idx, dim in zip(reversed(new_indices), reversed(dims)):
      node = Loop(idx, dim, 1, node)
    lowered_graph.append(node)
  return lowered_graph

def get_children_dfs(node:Node, visited:dict[Node, None]):
  if node in visited: return
  for source in node.sources:
    get_children_dfs(source, visited)
  visited.update(dict.fromkeys(node.sources, None))

def linearize(ast:list[Node]) -> set[Node]:
  visited:dict[Node, None] = {}
  for node in ast:
    node.terminal = True
    get_children_dfs(node, visited)
    visited[node] = None
  visited = list(visited)
  if DEBUG >= 2: print_procedure(visited)
  return visited

def parse_ast(ast:list[Node]):
  if DEBUG:
    print("GRAPH")
    print_graph(ast)
  ast = [ast] if isinstance(ast, Node) else ast
  ast = rewrite_arrays(ast)
  ast = lower_graph(ast)
  if DEBUG >= 2:
    print("LOWERED GRAPH")
    print_graph(ast)
  return linearize(ast)