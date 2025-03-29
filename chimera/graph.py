from __future__ import annotations
from typing import Callable
from chimera.dtype import DType, dtypes
from chimera.helpers import fully_flatten, get_shape, all_same, tupled
import inspect

### Graph Nodes ###

class Node():
  terminal:bool = False
  @property
  def sources(self) -> tuple[Node]: return ()
  @property
  def dtype(self) -> DType: return dtypes.void
  @property
  def shape(self) -> tuple[int]: return ()

  def print_tree(self):
    string = self.__repr__() + "\n"
    var_count = []
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
    self.value = value
  @property
  def dtype(self): return dtypes.python_to_dtype[type(self.value)]
  @property
  def shape(self): return ()
  def __repr__(self): return f"Const {self.value}"

class Var(Node):
  def __init__(self, data:Node, name:str="var"):
    self.data = data if isinstance(data, Node) else Const(data)
    self.name = name
  @property
  def sources(self): return (self.data,)
  @sources.setter
  def sources(self, sources): self.data = sources[0]
  @property
  def dtype(self): return self.data.dtype
  @property
  def shape(self): return self.data.shape
  def __repr__(self): return self.name

class Assign(Node):
  def __init__(self, var:Var):
    self.var = var
  @property
  def sources(self): return (self.var,)
  @sources.setter
  def sources(self, sources): self.var = sources[0]

class SymbolicIndex(Node):
  def __init__(self, name: str):
    self.name = name
  @property
  def dtype(self) -> type: dtypes.int32
  @property
  def shape(self): return (1,)
  def __repr__(self): return f"Idx {self.name}"

class Array(Node):
  def __init__(self, data):
    self._shape = get_shape(data)
    self.data = fully_flatten(data)
    assert all_same([type(d) for d in data]), f"Array must contain only one type but got {data}"
    self._dtype = dtypes.python_to_dtype[type(data[0])]
  @property
  def dtype(self): return self._dtype
  @property
  def shape(self): return self._shape
  def __len__(self): return self._shape[0]

class Index(Node):
  def __init__(self, data:Node, indices):
    self.data = data
    self.indices = tupled(indices)
    assert len(self.indices) <= len(self.data.shape), f"Too many indices {indices} for {data.shape}"
  @property
  def sources(self): return (self.data,) + self.indices
  @sources.setter
  def sources(self, sources):
    self.data, *self.indices = sources
    self.indices = tuple(self.indices)
  @property
  def dtype(self): return self.data.dtype
  @property
  def shape(self): return self.data.shape[len(self.indices):]
  def __repr__(self): return f"Index"

class Loop(Node):
  def __init__(self, start, stop, step, scope, idx=None):
    if not isinstance(start, Node): start = Const(start)
    self.stop = stop if isinstance(stop, Node) else Const(stop)
    self.step = step if isinstance(step, Node) else Const(step)
    self.scope = scope
    self.idx = idx if idx else Var(start, "idx")
    self.assign = Assign(idx)
  @property
  def sources(self): return (self.assign, self.stop, self.step, self.scope)
  @sources.setter
  def sources(self, sources): self.stop, self.step, self.assign, self.scope = sources
  @property
  def dtype(self): return self.scope.dtype
  @property
  def dtype(self): return self.scope.shape

class BinaryOp(Node):
  def __init__(self, op, left, right):
    self.op, self.left, self.right = op, left, right
  @property
  def sources(self): return (self.left, self.right)
  @sources.setter
  def sources(self, sources): self.left, self.right = sources
  @property
  def dtype(self): return self.left.dtype
  @property
  def shape(self):
    if len(self.left.shape) > len(self.right.shape): return self.left.shape
    if len(self.left.shape) < len(self.right.shape): return self.right.shape
    # Need to broadcast in the future
    return tuple(max(l, r) for l, r in zip(self.left.shape, self.right.shape))
  def __repr__(self): return f"{super().__repr__()} {self.op}"

class Function(Node):
  def __init__(self, args, body):
    self.args = (args,) if isinstance(args, Node) else args
    self.body = body
  @property
  def sources(self): return self.args + (self.body,)
  @sources.setter
  def sources(self, sources): self.args, self.body = sources
  @property
  def dtype(self): return self.body.dtype
  @property
  def dtype(self): return self.body.shape

class Call(Node):
  def __init__(self, func:Function, args:tuple[Node]):
    self.func = func
    self.args = (args,) if isinstance(args, Node) else args
  @property
  def sources(self): return (self.func) + self.args
  @sources.setter
  def sources(self, sources): self.func, self.args = sources
  @property
  def dtype(self): return self.func.dtype
  @property
  def shape(self): return self.func.shape

class Print(Node):
  def __init__(self, data):
    self.data = data
  @property
  def sources(self): return (self.data,)
  @sources.setter
  def sources(self, sources): self.data = sources[0]
  @property
  def dtype(self): return self.data.dtype
  @property
  def shape(self): return self.data.shape

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
    for i in range(len(node.sources)):
      source = node.sources[i]
      if isinstance(source, Array):
        if source in assignments:
          var = assignments[source].var
        else:
          var = Var(source, "arr")
          assign = Assign(var)
          assignments[source] = assign

        s = list(node.sources)
        s[i] = var
        node.sources = tuple(s)
    rewrite_arrays(node.sources, assignments)
  if first_call: return list(assignments.values()) + graph

# Propogates an Index node down the graph if its children are not data nodes (Array / Var)
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
      
  current.sources = tuple(propagate_indexing(index, s) for s in current.sources)
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
    shape = node.shape
    if isinstance(node, (Array, Var)):
      node = Index(node, indices + new_indices)
    elif node.sources:
      node.sources = lower_graph(node.sources, indices + new_indices)
    for idx, dim in zip(new_indices, shape):
      node = Loop(0, dim, 1, node, idx)
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
  return list(visited)

def parse_ast(ast:list[Node]):
  ast = [ast] if isinstance(ast, Node) else ast
  ast = rewrite_arrays(ast)
  print("REWRITE ARRAYS")
  for node in ast: node.print_tree()
  ast = lower_graph(ast)
  print("LOWER")
  for node in ast: node.print_tree()
  return linearize(ast)