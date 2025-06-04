from __future__ import annotations
from chimera.dtype import DType, dtypes
from chimera.helpers import fully_flatten, get_shape, all_same, listed, tupled, prod, strides_for_shape
from chimera.view import View
from dataclasses import dataclass
import weakref

class NodeMetaClass(type):
  node_cache:dict[tuple, weakref.ReferenceType[Node]] = {}
  def __call__(cls, *args, **kwargs):
    node:Node = super().__call__(*args, **kwargs)
    node._sources = getattr(node, "_sources", ())
    node._arg = getattr(node, "_arg", None)
    node._dtype = getattr(node, "_dtype", dtypes.void)
    node._view = getattr(node, "_view", View.create())
    node._shape = getattr(node, "_shape", ())
    return NodeMetaClass.register(node)
  @classmethod
  def register(cls, node:Node) -> Node:
    key = (type(node), node._sources, node._arg, node._dtype, node._shape)
    if not isinstance(node, Var) \
       and (wret := NodeMetaClass.node_cache.get(key, None)) is not None \
       and (ret := wret()) is not None:
      return ret
    NodeMetaClass.node_cache[key] = weakref.ref(node)
    return node

@dataclass(eq=False, slots=True)
class Node(metaclass=NodeMetaClass):
  _sources:tuple[Node]
  _arg:any
  _dtype:DType
  _view:View
  _shape:tuple[int]
  @property
  def sources(self) -> tuple[Node]: return self._sources
  @property
  def dtype(self) -> DType: return self._dtype
  @property
  def view(self) -> View: return self._view
  @property
  def shape(self) -> tuple[int, ...]: return self._shape

  def copy(self, sources:list[Node]=None, arg:any=None, dtype:DType=None, view:View=None, shape:tuple[int]=None) -> Node:
    new_node:Node = self.__class__.__new__(self.__class__)
    new_node._sources = tupled(sources) if sources is not None else self._sources
    new_node._arg = arg if arg is not None else self._arg
    new_node._dtype = dtype if dtype is not None else self._dtype
    new_node._view = view if view is not None else self._view
    new_node._shape = shape if shape is not None else self._shape
    new_node = NodeMetaClass.register(new_node)
    assert not self.sources or len(self.sources) == len(new_node.sources), f"New sources must be of same length as original: Expected {len(self._sources)}, Actual {len(new_node.sources)}.\nOriginal node: {self}"
    return new_node

  def print_tree(self, visited:set[Node]=None, var_count:list[Node]=None): print(self.get_print_tree(visited, var_count))
  def get_print_tree(self, visited:set[Node]=None, var_count:list[Node]=None, format_func:callable[str, Node]=None) -> str:
    if var_count is None: var_count = []
    if visited is None: visited = set()
    if format_func is None: format_func = lambda x: x.__repr__()
    string = self.__repr__() + "\n"
    for i, source in enumerate(self.sources): string += source._get_print_tree(visited, var_count, format_func, i == len(self.sources) - 1)
    return string
  def _get_print_tree(self, visited:set[Node], var_count:list[Node], format_func:callable[str, Node], is_last:bool, indent:str = ''):
    node_str = format_func(self)
    if isinstance(self, Var):
      if self not in var_count: var_count.append(self)
      node_str += f" {var_count.index(self)}"
    tree_str = indent + ("└─" if is_last else "├─") + node_str + "\n"
    indent += "  " if is_last else "│ "
    if self in visited: return tree_str
    visited.add(self)
    for i, source in enumerate(self.sources): tree_str += source._get_print_tree(visited, var_count, format_func, i == len(self.sources) - 1, indent)
    return tree_str

  @staticmethod
  def to_node(x): return x if isinstance(x, Node) else Const(x)

  def __repr__(self): return self.__class__.__name__
  def __add__(self, x): return BinaryOp('+', self, Node.to_node(x))
  def __sub__(self, x): return BinaryOp('-', self, Node.to_node(x))
  def __mul__(self, x): return BinaryOp('*', self, Node.to_node(x))
  def __mod__(self, x): return BinaryOp('%', self, Node.to_node(x))
  def __truediv__(self, x): return BinaryOp('/', self, Node.to_node(x))
  def __getitem__(self, key): return Index(self, key)

class Program(Node):
  def __init__(self, nodes:list[Node]|Node):
    self._sources = tupled(nodes)

class Const(Node):
  def __init__(self, value):
    assert isinstance(value, (int, float)), f"Const node can only have values of type int, float. Type was {type(value)}. {value}"
    self._arg = value
    self._dtype = dtypes.python_to_dtype[type(self.value)]
    self._shape = ()
  @property
  def value(self): return self._arg
  def __repr__(self): return f"Const {self.value}"

class Var(Node):
  def __init__(self, data:Allocate|Array|Const, name:str="var"):
    assert data != None, "data should not be none"
    data = Node.to_node(data)
    assert isinstance(data, (Allocate, Array, Const)), f"data is not valid type. {data}"
    self._sources = (data,)
    self._view = self.data.view
    self._shape = self.data._shape
    self._dtype = self.data.dtype
    self._arg = (name, data.strides if isinstance(data, Array) else strides_for_shape(data.shape))
  @property
  def name(self) -> str: return self._arg[0]
  @property
  def strides(self) -> Node: return self._arg[1]
  @property
  def data(self) -> Node: return self.sources[0]
  def __repr__(self): return self.name

class Assign(Node):
  def __init__(self, var:Var):
    self._sources = (var,)
  @property
  def var(self) -> Var: return self.sources[0]

class Allocate(Node):
  def __init__(self, shape:tuple[int], dtype:DType):
    self._arg = dtype.itemsize * prod(shape)
    self._dtype = dtype
    # self._view = data.view
    self._shape = shape
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
    self._shape = data.shape
    self._dtype = data.dtype
  @property
  def data(self) -> Node: return self.sources[0]
  @property
  def value(self) -> Node: return self.sources[1]

class Array(Node):
  def __init__(self, data:list):
    self._view = View.create(get_shape(data))
    self._shape = get_shape(data)
    self._arg = (tuple(fully_flatten(data)), self._view.strides)
    assert all_same([type(d) for d in data]), f"Array must contain only one type but got {data}"
    self._dtype = dtypes.python_to_dtype[type(self.data[0])]
  @property
  def data(self) -> tuple: return self._arg[0]
  @property
  def strides(self) -> int: return self._arg[1]

class Index(Node):
  def __init__(self, data:Node, indices:int|Const|list[int|Const]):
    assert isinstance(indices, (int, list, tuple)) or (isinstance(indices, Const|Var) and indices.dtype is dtypes.int32), f"Expected indices to be int, Const, Var or list thereof but got {indices}"
    indices = tuple(map(Node.to_node, listed(indices)))
    assert data.shape != (), f"Cannot index node with no shape {data}"
    assert len(indices) > 0, f"Cannot index {data} with no indices"
    strides = data.view.strides[:len(data.shape) - len(indices)]
    self._sources = (data, *indices)
    self._view = View.create(self.data.shape[len(indices):], strides)
    self._shape = self.data.shape[len(indices):]
    self._dtype = self.data.dtype
  @property
  def data(self) -> Node: return self.sources[0]
  @property
  def indices(self) -> Node: return self.sources[1:]

class Load(Node):
  def __init__(self, data:Var, indices:int|Const|list[int|Const]):
    assert isinstance(data, Var), "Only variables can be loaded"
    assert len(data.strides) == len(indices), f"Indices has to match data shape.\nIndices: {indices}\nStrides: {data.strides}\nNode: {data}"
    indexer = Const(0)
    for idx, stride in zip(reversed(indices), data.strides):
      indexer = indexer + idx * stride
    self._sources = (data, indexer)
  @property
  def data(self) -> Node: return self.sources[0]
  @property
  def indexer(self) -> Node: return self.sources[1]

class Loop(Node):
  def __init__(self, start:Var|Const, stop:Const, scope:Node):
    if not isinstance(start, Var): start = Var(Node.to_node(start), "idx")
    self._sources = (Assign(start), Node.to_node(stop), scope)
    self._view = scope.view
    self._shape = scope.shape
    assert isinstance(self.stop, Const), f"Loop stop must be a constant but got {type(self.stop)}"
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
    self._sources = tuple(s if s.view == v else Expand(s, v.shape) for s,v in zip(self.sources, self._broadcast_views(views)))
    self._view = self.left.view if len(self.left.shape) >= len(self.right.shape) else self.right.view
    self._shape = self.left.shape if len(self.left.shape) >= len(self.right.shape) else self.right.shape
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
  def __init__(self, node:Node, shape:tuple[int, ...]):
    assert len(node.shape) <= len(shape), f"Expand has to have same or more dimensions as the source node\nSource shape: {node.shape}\nExpand shape: {shape}"
    assert all(s1 == 1 or s1 == s2 for s1, s2 in zip(node.shape, shape[len(shape) - len(node.shape):])),\
    f"Expanded shape is invalid for source shape\nSource shape: {' '*(len(str(shape))-len(str(node.shape)))}{node.shape}\nExpand shape: {shape}"
    self._sources = (node,)
    self._shape = shape
    self._dtype = self.node.dtype
  @property
  def node(self) -> Node: return self.sources[0]    

class Reshape(Node):
  def __init__(self, node:Node, shape:tuple[int, ...]):
    assert prod(node.shape) == prod(shape), f"Cannot reshape {node.shape} to {shape} as they differ in size ({prod(node.shape)}, {prod(shape)})"
    self._sources = (node,)
    # self._view = View.create(shape) # TODO: take previous view into account
    self._shape = shape
    self._dtype = self.node.dtype
  @property
  def node(self) -> Node: return self.sources[0]

class Debug(Node):
  def __init__(self, data:Node):
    self._sources = (data,)
    self._view = data.view
    self._shape = data.shape
    self._dtype = data.dtype
  @property
  def data(self): return self.sources[0]