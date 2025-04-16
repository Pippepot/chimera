from __future__ import annotations
from typing import Callable, TypeVar
from chimera.dtype import DType, dtypes
from chimera.helpers import DEBUG, fully_flatten, get_shape, all_same, all_instance, listed, tupled, prod
from chimera.view import View
import inspect

def print_graph(nodes:list[Node]):
  visited, var_count = set(), []
  for n in nodes: n.print_tree(visited, var_count)

def print_procedure(nodes:list[Node]):
  for i,n in enumerate(nodes):
    formatted_parents = [nodes.index(x) if x in nodes else "--" for x in n.sources]
    print(f"{i:4d} {str(n):20s}: {str(n.dtype.name):8s} {str(formatted_parents):16s}")

### Graph Nodes ###

class Node():
  terminal:bool = False
  _sources:tuple[Node] = ()
  _arg:any = None
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

  def copy(self, sources:list[Node]=None, arg:any=None, dtype:DType=None, view:View=None) -> Node:
    new_node:Node = self.__new__(self.__class__)
    new_node._sources = tupled(sources) if sources is not None else self._sources
    new_node._arg = arg if arg is not None else self._arg
    new_node._dtype = dtype if dtype is not None else self._dtype
    new_node._view = view if view is not None else self._view
    assert not self.sources or len(self.sources) == len(new_node.sources), f"New sources must be of same length as original: Expected {len(self._sources)}, Actual {len(new_node.sources)}"
    return new_node

  def print_tree(self, visited:set[Node]=None, var_count:list[Node]=None):
    if var_count is None: var_count = []
    if visited is None: visited = set()
    string = self.__repr__() + "\n"
    for i, source in enumerate(self.sources): string += source._get_print_tree(visited, var_count, i == len(self.sources) - 1)
    print(string)

  def _get_print_tree(self, visited:set[Node], var_count:list[Node], is_last:bool, indent:str = ''):
    node_str = self.__repr__()
    if isinstance(self, Var):
      if self not in var_count: var_count.append(self)
      node_str += f" {var_count.index(self)}"
    tree_str = indent + ("└─" if is_last else "├─") + node_str + "\n"
    indent += "  " if is_last else "│ "
    if self in visited: return tree_str
    visited.add(self)
    for i, source in enumerate(self.sources): tree_str += source._get_print_tree(visited, var_count, i == len(self.sources) - 1, indent)
    return tree_str

  @staticmethod
  def to_node(x): return x if isinstance(x, Node) else Const(x)

  def __repr__(self): return self.__class__.__name__
  def __add__(self, x): return BinaryOp('+', self, Node.to_node(x))
  def __sub__(self, x): return BinaryOp('-', self, Node.to_node(x))
  def __mul__(self, x): return BinaryOp('*', self, Node.to_node(x))
  def __truediv__(self, x): return BinaryOp('/', self, Node.to_node(x))
  def __getitem__(self, key): return Index(self, key)

class Const(Node):
  def __init__(self, value):
    assert isinstance(value, (int, float)), f"Const node can only have values of type int, float. Type was {type(value)}"
    self._arg = value
    self._dtype = dtypes.python_to_dtype[type(self.value)]
  @property
  def value(self): return self._arg
  def __repr__(self): return f"Const {self.value}"

class Var(Node):
  def __init__(self, data:Node, name:str="var"):
    assert data != None, "data should not be none"
    self._sources = (Node.to_node(data),)
    self._view = self.data.view
    self._dtype = self.data.dtype
    self._arg = name
  @property
  def name(self) -> str: return self._arg
  @property
  def data(self) -> Node: return self.sources[0]
  def __repr__(self): return self.name

class Assign(Node):
  def __init__(self, var:Var):
    self._sources = (var,)
  @property
  def var(self) -> Var: return self.sources[0]

class Allocate(Node):
  def __init__(self, data:Node):
    self._arg = data.dtype.itemsize * prod(data.shape)
    self._dtype = data.dtype
    self._view = data.view
  @property
  def size(self) -> int: return self._arg

class Free(Node):
  def __init__(self, var:Var):
    self._sources = (var,)
  @property
  def var(self) -> Var: return self.sources[0]

class Store(Node):
  def __init__(self, data:Node, value:Node):
    self._sources = (data, value)
    self._view = data.view
    self._dtype = data.dtype
  @property
  def data(self) -> Node: return self.sources[0]
  @property
  def value(self) -> Node: return self.sources[1]

class Array(Node):
  def __init__(self, data:list):
    self._view = View.create(get_shape(data))
    self._arg = fully_flatten(data)
    assert all_same([type(d) for d in data]), f"Array must contain only one type but got {data}"
    self._dtype = dtypes.python_to_dtype[type(self.data[0])]
  @property
  def data(self) -> list: return self._arg

class Index(Node):
  def __init__(self, data:Node, indices:int|list[int]):
    self._sources = (data,) + tuple(map(Node.to_node, listed(indices)))
    self._view = View.create(self.data.shape[len(self.indices):]) #TODO probs not correct
    self._dtype = self.data.dtype
  @property
  def data(self) -> Node: return self.sources[0]
  @property
  def indices(self) -> Node: return self.sources[1:]

class Loop(Node):
  def __init__(self, start:Var|Const, stop:Const, scope:Node):
    if not isinstance(start, Var): start = Var(Node.to_node(start), "idx")
    self._sources = (Assign(start), Node.to_node(stop), scope)
    self._view = self.scope.view
  @property
  def assign(self) -> Assign: return self.sources[0]
  @property
  def idx(self) -> Var: return self.assign.var
  @property
  def stop(self) -> Const: return self.sources[1]
  @property
  def scope(self) -> Node: return self.sources[2]

class BinaryOp(Node):
  def __init__(self, op, left:Node, right:Node):
    self._sources = (left, right)
    self._arg = op

    views = [self.left.view, self.right.view]
    self._sources = tuple(s if s.view == v else Expand(s, v) for s,v in zip(self.sources, self._broadcast_views(views)))
    self._view = self.left.view
    self._dtype = self.left.dtype # TODO resolve better

  @property
  def op(self): return self._arg
  @property
  def left(self): return self.sources[0]
  @property
  def right(self): return self.sources[1]
  def _broadcast_views(self, views_in:tuple[View]) -> tuple[View]:
    empty_idx:list[int] = []
    views:list[View] = []
    for i,view in enumerate(views_in):
      if view.shape == (): empty_idx.append(i)
      else: views.append(view)
    if not len(views): return views_in

    max_shape = max(len(view.shape) for view in views)
    # Left align shapes
    shapes = [(1,)*(max_shape - len(view.shape)) + view.shape for view in views]
    strides = [(0,)*(max_shape - len(view.strides)) + view.strides for view in views]
    new_shape = tuple(max(shape[i] for shape in shapes) for i in range(max_shape))
    assert all(all(dim == 1 or dim == new_dim for dim,new_dim in zip(shape, new_shape)) for shape in shapes),\
      f"Cannot broadcast shapes {(v.shape for v in views)}"
    views = [View.create(new_shape, stride) for stride in strides]
    for i in empty_idx: views.insert(i, View.create())
    return tuple(views)
  def __repr__(self): return f"{super().__repr__()} {self.op}"

class Expand(Node):
  def __init__(self, node:Node, view:View):
    #TODO: assert that the view is compatible with the node
    self._sources = (node,)
    self._view = view
    self._dtype = self.node.dtype
  @property
  def node(self) -> Node: return self.sources[0]    

class Reshape(Node):
  def __init__(self, node:Node, shape:tuple[int, ...]):
    assert prod(node.shape) == prod(shape), f"Cannot reshape {node.shape} to {shape} as they differ in size ({prod(node.shape)}, {prod(shape)})"
    self._sources = (node,)
    self._view = View.create(shape) # TODO: take previous view into account
    self._dtype = self.node.dtype
  @property
  def node(self) -> Node: return self.sources[0]
    

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
  def __init__(self, data:Node):
    self._sources = (data,)
  @property
  def data(self): return self.sources[0]

### Pattern Matching ###

T = TypeVar('T', bound=Node)

class Pat():
  __slots__ = ["type", "predicate", "name", "sources", "fuzzy_source_match"]
  def __init__(self, type:type[T]|tuple[type[T]]=None, predicate:Callable[[T], bool]=None,
               name:str=None, sources:Pat|tuple[Pat]=None, fuzzy_source_match:bool=False):
    if type is None: type = (Node,)
    self.type:tuple[type] = type if isinstance(type, tuple) else (type,)
    self.predicate = predicate
    self.name = name
    self.sources:tuple[Pat] = sources if isinstance(sources, tuple) or sources is None else (sources,)
    self.fuzzy_source_match = fuzzy_source_match
  def match(self, node:Node, args:dict[str, Node|tuple[Node]]) -> bool:
    if (not any(isinstance(node, t) for t in self.type)) or \
     (self.predicate is not None and not self.predicate(node)): return False
    if self.sources is not None:
      if self.fuzzy_source_match:
        if len(self.sources) > len(node.sources): return False
        if not all(pat.match(source, args) for pat, source in zip(self.sources, node.sources[:len(self.sources)])): return False
      else:
        if len(self.sources) != len(node.sources): return False
        if not all(pat.match(source, args) for pat, source in zip(self.sources, node.sources)): return False
    if self.name is not None: args[self.name] = node
    return True
  def __repr__(self):
    return f"Pat({self.type}, {self.name})"

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
def propagate_index(idx:Index) -> Node:
   sources = map(lambda x: x if x.shape == () else Index(x, idx.indices), idx.data.sources)
   n = idx.data.copy(sources, idx.data._arg, idx.data.dtype, idx.view)
   return n
def rewrite_index(idx: Index, reshape: Reshape) -> Index:
    print(f"REWRITE: {idx} -> {reshape}")
    # Compute the flat index from idx.indices in the new (reshaped) space.
    flat_index = Const(0)
    for index_node, stride in zip(idx.indices, reshape.view.strides):
        term = BinaryOp("*", index_node, Const(stride))
        flat_index = BinaryOp("+", flat_index, term)

    # Convert the flat index to multi-dimensional indices for the original shape.
    new_indices = []
    remaining = flat_index
    for stride in reshape.node.view.strides:
        quotient = BinaryOp("/", remaining, Const(stride))
        new_indices.append(quotient)
        remaining = BinaryOp("%", remaining, Const(stride))
    
    # Return an Index node using the original node and the new multi-dimensional indices.
    return Index(reshape.node, new_indices)

base_rewrite = PatternMatcher([
  # Refactor print statements to use malloc/free
  (Pat(Print, predicate=lambda x: x.data.shape != (), name="x"), refactor_print),
  # Propagate indexing down the graph
  (Pat(Index, name="idx", sources=(Pat(Reshape, name="reshape")), fuzzy_source_match=True), rewrite_index),
  (Pat(Index, name="idx", sources=(Pat((BinaryOp, Store))), fuzzy_source_match=True), propagate_index),
  (Pat(Index, name="idx1", sources=(Pat(Index, name="idx2")), fuzzy_source_match=True), lambda idx1,idx2: Index(idx2.data, idx1.indices + idx2.indices)),
  # Create ranges to lower indices in the graph
  # (Pat((BinaryOp, Store), predicate=lambda x: x.shape != (), name="x"), lambda x: create_loop(x)),
  # Move array to assignments
  (Pat((Index, BinaryOp, Expand, Reshape), name="parent"), assign_array),
])

def rewrite_graph(n:Node, rewriter:PatternMatcher, ctx=None, replace:dict[Node, Node]=None) -> Node:
  if replace is None: replace = {}
  elif (rn := replace.get(n)) is not None: return rn
  new_n:Node = n
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
    node.terminal = True
    _get_children_dfs(node, visited)
    visited[node] = None
  visited = list(visited)
  if DEBUG >= 2: print_procedure(visited)
  return visited

def parse_ast(ast:list[Node]):
  if DEBUG:
    print("GRAPH")
    print_graph(ast)    
  ast = rewrite_graph(ast[0], base_rewrite, (context:=RewriteContext()))
  ast = list(context.pre.values()) + [ast] + list(context.post.values()) 
  if DEBUG >= 2:
    print("LOWERED GRAPH")
    print_graph(ast)
  return linearize(ast)