from __future__ import annotations
from chimera.dtype import DType, dtypes
from chimera.helpers import LOG_SHAPES, fully_flatten, get_shape, all_same, listed, tupled
from dataclasses import dataclass
from enum import auto, IntEnum, Enum
import weakref, functools, math

class Ops(IntEnum):
  # Unary ops
  SQRT = auto(); LOG2 = auto(); EXP2 = auto(); SIN = auto()
  # Binary ops
  ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MOD = auto(); MAX = auto()
  SHL = auto(); SHR = auto(); AND = auto(); OR = auto(); XOR = auto(); POW = auto()
  CMPLT = auto(); CMPNE = auto()
  def __str__(self): return Enum.__str__(self)

class NodeMetaClass(type):
  node_cache:dict[tuple, weakref.ReferenceType[Node]] = {}
  def __call__(cls, *args, **kwargs):
    node:Node = super().__call__(*args, **kwargs)
    node._sources = getattr(node, "_sources", ())
    node._arg = getattr(node, "_arg", None)
    node._dtype = getattr(node, "_dtype", dtypes.void)
    node._shape = getattr(node, "_shape", ())
    return NodeMetaClass.register(node)
  @classmethod
  def register(cls, node:Node) -> Node:
    key = (type(node), node._sources, node._arg, node._dtype, node._shape)
    if not isinstance(node, (Var, ScopeBegin, ScopeEnd)) \
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
  _shape:tuple[Const|Var|BinaryOp]
  @property
  def sources(self) -> tuple[Node]: return self._sources
  @property
  def dtype(self) -> DType: return self._dtype
  @property
  def shape(self) -> tuple[Const|Var|BinaryOp, ...]: return self._shape

  def copy(self, sources:list[Node]=None, arg:any=None, dtype:DType=None, shape:tuple[Const|Var|BinaryOp]=None) -> Node:
    new_node:Node = self.__class__.__new__(self.__class__)
    new_node._sources = tupled(sources) if sources is not None else self._sources
    new_node._arg = arg if arg is not None else self._arg
    new_node._dtype = dtype if dtype is not None else self._dtype
    new_node._shape = shape if shape is not None else self._shape
    new_node = NodeMetaClass.register(new_node)
    assert not self.sources or len(self.sources) == len(new_node.sources), f"New sources must be of same length as original: Expected {len(self._sources)}, Actual {len(new_node.sources)}.\nOriginal node: {self}"
    return new_node

  def print_tree(self, visited:set[Node]=None, var_count:list[Node]=None): print(self.get_print_tree(visited, var_count))
  def get_print_tree(self, visited:set[Node]=None, var_count:list[Node]=None, format_func:callable[str, Node]=None) -> str:
    if var_count is None: var_count = []
    if visited is None: visited = set()
    if format_func is None: format_func = lambda x: x.__repr__()
    string = self.__repr__() + f" shape={self.shape}\n" if LOG_SHAPES else "\n"
    for i, source in enumerate(self.sources): string += source._get_print_tree(visited, var_count, format_func, i == len(self.sources) - 1)
    return string
  def _get_print_tree(self, visited:set[Node], var_count:list[Node], format_func:callable[str, Node], is_last:bool, indent:str = ''):
    node_str = format_func(self)
    if isinstance(self, Var):
      if self not in var_count: var_count.append(self)
      node_str += f" {var_count.index(self)}"
    if LOG_SHAPES: node_str += f" shape={self.shape}"
    tree_str = indent + ("└─" if is_last else "├─") + node_str + "\n"
    indent += "  " if is_last else "│ "
    if self in visited: return tree_str
    visited.add(self)
    for i, source in enumerate(self.sources): tree_str += source._get_print_tree(visited, var_count, format_func, i == len(self.sources) - 1, indent)
    return tree_str

  def simplify(self):
    if isinstance(self, Const): return self
    from chimera.symbolic import symbolic
    from chimera.rewrite import rewrite_tree
    return rewrite_tree(self, symbolic, track_rewrites=False)

  @staticmethod
  def to_node(x): return x if isinstance(x, Node) else Const(x)
  @functools.cached_property
  def strides(self) -> tuple[Node]:
    strides = []
    stride = Const(1)
    for s in self.shape[::-1]:
      simp = s.simplify()
      if simp == Const(1):
        strides.append(Const(0))
        continue
      strides.append(stride)
      stride *= simp
    return tuple(reversed(strides))

  def logical_not(self): return self.ne(True)
  def _binop(self, op, x, reverse=False):
    a, b = broadcast(x, self) if reverse else broadcast(self, x)
    return BinaryOp(op, a, b)
  
  def __repr__(self): return self.__class__.__name__
  def __add__(self, x): return self._binop(Ops.ADD, x)
  def __sub__(self, x): return self._binop(Ops.SUB, x)
  def __mul__(self, x): return self._binop(Ops.MUL, x)
  def __mod__(self, x): return self._binop(Ops.MOD, x)
  def __truediv__(self, x): return self._binop(Ops.DIV, x)
  def __lshift__(self, x): return self._binop(Ops.SHL, x)
  def __rshift__(self, x): return self._binop(Ops.SHR, x)
  def __pow__(self, x): return self._binop(Ops.POW, x)
  def __and__(self, x): return self._binop(Ops.AND, x)
  def __or__(self, x): return self._binop(Ops.OR, x)
  def __xor__(self, x): return self._binop(Ops.XOR, x)

  def __radd__(self, x): return self._binop(Ops.ADD, x, True)
  def __rsub__(self, x): return self._binop(Ops.SUB, x, True)
  def __rmul__(self, x): return self._binop(Ops.MUL, x, True)
  def __rmod__(self, x): return self._binop(Ops.MOD, x, True)
  def __rtruediv__(self, x): return self._binop(Ops.DIV, x, True)
  def __rlshift__(self, x): return self._binop(Ops.SHL, x, True)
  def __rrshift__(self, x): return self._binop(Ops.SHR, x, True)
  def __rpow__(self, x): return self._binop(Ops.POW, x, True)
  def __rand__(self, x): return self._binop(Ops.AND, x, True)
  def __ror__(self, x): return self._binop(Ops.OR, x, True)
  def __rxor__(self, x): return self._binop(Ops.XOR, x, True)

  def __lt__(self, x): return self._binop(Ops.CMPLT, x)
  def __gt__(self, x): return self._binop(Ops.CMPLT, x, True)
  def __ge__(self, x): return (self < x).logical_not()
  def __le__(self, x): return (self > x).logical_not()
  
  def __neg__(self): return self*(-1)

  def ne(self, x): return self._binop(Ops.CMPNE, x)
  def eq(self, x): return self.ne(x).logical_not()
  # def sqrt(self): return self.alu(Ops.SQRT)
  # def sin(self): return self.alu(Ops.SIN)
  # def log2(self): return self.alu(Ops.LOG2)
  # def exp2(self): return self.alu(Ops.EXP2)
  def pow(self, x): return self._binop(Ops.POW, x)
  def where(self, passed, failed):
    passed, failed = broadcast(passed, failed)
    cond, passed = broadcast(self, passed)
    cond, failed = broadcast(cond, failed)
    return Where(cond, passed, failed)

class ScopeBegin(Node):
  def __init__(self): pass
class ScopeEnd(Node):
  def __init__(self): pass

class Const(Node):
  def __init__(self, value):
    assert isinstance(value, (int, float, bool)), f"Const node can only have values of type int, float, bool. Type was {type(value)}. {value}"
    self._arg = int(value) if isinstance(value, bool) else value
    self._dtype = dtypes.get_dtype(value)
  @property
  def value(self): return self._arg
  def __repr__(self): return f"Const {self.value}"

class Var(Node):
  def __init__(self, data:Function|Allocate|Array|Const|Reshape|Expand, name:str="var"):
    assert data != None, "data should not be none"
    data = Node.to_node(data)
    assert isinstance(data, (Function, Allocate, Array, Const, Reshape, Expand)), f"data is not valid type. {data}"
    self._sources = (data,)
    self._shape = self.data.shape
    self._dtype = self.data.dtype
    self._arg = name
  @property
  def name(self) -> str: return self._arg
  @property
  def data(self) -> Node: return self.sources[0]
  def __repr__(self): return self.name

class Allocate(Node):
  def __init__(self, shape:Node, dtype:DType):
    shape = shape.simplify()
    self._sources = ((shape * dtype.itemsize).simplify(),)
    self._shape = (shape,)
    self._dtype = dtype
  @property
  def size(self) -> Node: return self.sources[0]

class Free(Node):
  def __init__(self, var:Var):
    self._sources = (var,)
  @property
  def var(self) -> Var: return self.sources[0]

class Store(Node):
  def __init__(self, data:Node, value:Node):
    self._sources = (data, value)
    self._shape = data.shape
    self._dtype = data.dtype
  @property
  def data(self) -> Node: return self.sources[0]
  @property
  def value(self) -> Node: return self.sources[1]

class Array(Node):
  def __init__(self, data:list):
    self._shape = tuple(map(Node.to_node, get_shape(data)))
    self._arg = tuple(fully_flatten(data))
    assert all_same([type(d) for d in data]), f"Array must contain only one type but got {data}"
    self._dtype = dtypes.get_dtype(self.data[0])
    if self.dtype == dtypes.bool: self._arg = tuple(map(int, self._arg))
  @property
  def data(self) -> tuple: return self._arg

class Slice(Node):
  def __init__(self, begin:int|Const, end:int|Const, step:int|Const=Const(1)):
    begin, end, step = Node.to_node(begin), Node.to_node(end), Node.to_node(step)
    assert type(begin) == type(end) == type(step) == Const, f"Arguments has to be const.\nBegin: {begin}\End: {end}\Step: {step}"
    assert step != 0, "Step cannot be 0"
    assert math.copysign(step.value, end.value) == step.value, "Step and end cannot have different signs"
    self._sources = (begin, end, step)
    self._arg = (step > 0).where((end - begin + step - 1) / step, (end - begin - step - 1) / (-step)).simplify()
  @property
  def begin(self) -> Const: return self._sources[0]
  @property
  def step(self) -> Const: return self._sources[2]
  @property
  def length(self) -> BinaryOp: return self._arg

IndexerType = int|Const|Slice

class Index(Node):
  def __init__(self, data:Node, indices:IndexerType|list[IndexerType]):
    indices = listed(indices)
    assert data.shape != (), f"Cannot index node with no shape {data}"
    assert len(indices) > 0, f"Cannot index {data} with no indices"
    for i in indices:
      assert isinstance(i, (int, Slice)) or (isinstance(i, (Const, Var, BinaryOp)) and i.dtype is dtypes.int), f"Expected indices to be int, Const, Var, Slice or list thereof but got {indices}"
    indices = tuple(map(Node.to_node, indices))
    self._sources = (data, *indices)
    shape = []
    for i, s in enumerate(self.data.shape): 
      if i >= len(indices): shape.append(s)
      elif isinstance(indices[i], Slice): shape.append(indices[i].length)
    self._shape = tuple(shape)
    self._dtype = self.data.dtype
  @property
  def data(self) -> Node: return self.sources[0]
  @property
  def indices(self) -> Node: return self.sources[1:]

class Load(Node):
  def __init__(self, data:Var, indices:int|Const|list[int|Const]):
    assert isinstance(data, (Var, Call)), "Only variables can be loaded"
    assert len(data.strides) == len(indices), f"Indices has to match data shape.\nIndices: {indices}\nStrides: {data.strides}\nNode: {data}"
    indexer = Const(0)
    for idx, stride in zip(indices, data.strides):
      indexer = indexer + idx * stride
    self._sources = (data, indexer.simplify())
  @property
  def data(self) -> Node: return self.sources[0]
  @property
  def indexer(self) -> Node: return self.sources[1]

class Range(Node):
  def __init__(self, stop:Const|Var):
    self._sources = (stop,)
  def stop(self) -> Const: return self.sources[1]

class BinaryOp(Node):
  def __init__(self, op:Ops, left:Node, right:Node):
    assert left.shape == right.shape, f"Left and right operands need to have same shape\nLeft {left}: {left.shape}\nRight {right}: {right.shape}"
    self._arg = op
    self._sources = (left, right)
    self._shape = right.shape
    self._dtype = dtypes.bool if op in {Ops.CMPLT, Ops.CMPNE} else right.dtype
  @property
  def op(self): return self._arg
  @property
  def left(self): return self.sources[0]
  @property
  def right(self): return self.sources[1]
  def __repr__(self): return f"{super().__repr__()} {self.op}"

class Expand(Node):
  def __init__(self, node:Node, shape:tuple[int, ...]):
    assert len(node.shape) <= len(shape), f"Expand has to have same or more dimensions as the source node\nSource shape: {node.shape}\nExpand shape: {shape}"
    assert all(s1 == Const(1) or s1 == s2 for s1, s2 in zip(node.shape, shape[len(shape) - len(node.shape):])),\
    f"Expanded shape is invalid for source shape\nSource shape: {' '*(len(str(shape))-len(str(node.shape)))}{node.shape}\nExpand shape: {shape}"
    self._sources = (node,)
    self._shape = shape
    self._dtype = self.node.dtype
  @property
  def node(self) -> Node: return self.sources[0]    

class Reshape(Node):
  def __init__(self, node:Node, shape:tuple[int, ...]):
    self._sources = (node,)
    self._shape = tuple(self.to_node(s).simplify() for s in shape)
    self._dtype = self.node.dtype
  @property
  def node(self) -> Node: return self.sources[0]

class Permute(Node):
  def __init__(self, node:Node, dims:tuple[int, ...]):
    dims = tupled(dims)
    assert len(node.shape) == len(dims), f"Shape of {node}, {node.shape} does not have the same length as dims {dims}"
    assert all(i == a for i,a in enumerate(sorted(dims))), f"Invalid dims {dims} for shape {node.shape}"
    self._sources = (node,)
    self._shape = tuple(node.shape[i] for i in dims)
    self._dtype = node.dtype
    inv = [0] * len(dims)
    for d, pd in enumerate(dims): inv[pd] = d
    self._arg = tuple(inv)
  @property
  def node(self) -> Node: return self.sources[0]
  @property
  def inverse_permutation(self) -> tuple[int]: return self._arg

class Flip(Node):
  def __init__(self, node:Node, dims:tuple[int, ...]):
    dims = tupled(dims)
    assert_unique_dims(dims, node.shape)
    self._sources = (node,)
    self._shape = node.shape
    self._dtype = node.dtype 
    self._arg = dims
  @property
  def node(self) -> Node: return self.sources[0]
  @property
  def dims(self) -> tuple[int]: return self._arg

class Function(Node):
  def __init__(self, args:tuple[Var], name="func"):
    args = tupled(args)
    assert all(isinstance(a, Var) for a in args), f"Arguments to function must be Var, found {args}"
    self._sources = args
    self._arg = name
  @property
  def args(self) -> tuple[Node]: return self.sources[1:]
  @property
  def name(self) -> str: return self._arg

class Return(Node):
  def __init__(self, value:Node):
    assert value.dtype != dtypes.void, "Cannot return a statement"
    self._sources = (value,)
  @property
  def value(self) -> Node: return self.sources[0]

class Call(Node):
  def __init__(self, function:Function, args:tuple[Node]):
    assert isinstance(function, Function), f"Call has to call a function, found {function}"
    self._sources = (Var(function, function.name), *tupled(args))
    self._shape = function.shape
    self._dtype = function.dtype
  @property
  def function(self) -> Node: return self.sources[0]
  @property
  def args(self) -> tuple[Node]: return self.sources[1:]

class Reduce(Node):
  def __init__(self, value:Node, function:Function, dims:tuple[int, ...]):
    dims = tupled(dims)
    assert_unique_dims(dims, value.shape)
    assert function.dtype == value.dtype, f"Function has to return same type as the reduced value\nFunction type: {function.dtype}\nValue type: {value.dtype}"
    assert len(function.args), f"Function has to have exactly two arguments\nArgs: {function.args}"
    assert all(value.dtype == a.dtype for a in function.args), f"Both function argument types has to match the value's type\nArg types: {tuple(a.dtype for a in function.args)}\nValue type: {value.dtype}"
    self._sources = (value, function)
    self._shape = tuple(s for i, s in enumerate(value.shape) if i not in dims)
    assert function.shape == self.shape, f"Function shape {function.shape} has to equal shape {self.shape} given value's shape {value.shape} and dims {dims}"
    self._dtype = value.dtype
    self._arg = dims
  @property
  def value(self) -> Node: return self.sources[0]
  @property
  def function(self) -> Node: return self.sources[1]

class Where(Node):
  def __init__(self, condition:Node, passed:Node, failed:Node):
    assert condition.dtype == dtypes.bool, f"Condition has to be a bool\nCondition:\n{condition.get_print_tree()}"
    assert passed.shape == failed.shape, f"Expressions must be of same shape\nTrue: {passed.shape}\nFalse: {failed.shape}"
    assert passed.dtype == failed.dtype, f"Expressions must be of same data type\nTrue: {passed.dtype}\nFalse: {failed.dtype}"
    self._sources = (condition, passed, failed)
    self._shape = passed.shape
    self._dtype = passed.dtype
  @property
  def condition(self) -> Node: return self.sources[0]
  @property
  def passed(self) -> Node: return self.sources[1]
  @property
  def failed(self) -> Node: return self.sources[2]

class Branch(Node):
  def __init__(self, condition:Node, passed:Node, failed:Node|None):
    assert condition.dtype == dtypes.bool, f"Condition has to be a bool\nCondition:\n{condition.get_print_tree()}"
    self._sources = (condition, passed, failed)
  @property
  def condition(self) -> Node: return self.sources[0]
  @property
  def passed(self) -> Node: return self.sources[1]
  @property
  def failed(self) -> Node: return self.sources[2]

class Debug(Node):
  def __init__(self, data:Node):
    self._sources = (data,)
    self._shape = data.shape
    self._dtype = data.dtype
  @property
  def data(self): return self.sources[0]

class Block(Node):
  def __init__(self, *sources:Node):
    self._sources = tupled(sources)
    assert self._sources, "Sources cannot be empty"
    self._shape = self.expression.shape
    self._dtype = self.expression.dtype
  @property
  def expression(self) -> Node: return self.sources[-1]

def broadcast(left:Node, right:Node) -> tuple[Node, Node, tuple[int, ...]]:
  left, right = Node.to_node(left), Node.to_node(right)
  left_shape, right_shape = left.shape, right.shape
  if left_shape == right_shape: return left, right
  ls, rs = len(left.shape), len(right.shape)
  if ls < rs: left_shape = (Const(1),)*(rs-ls) + left_shape
  if rs < ls: right_shape = (Const(1),)*(ls-rs) + right_shape
  target_shape = []
  for l, r in zip(left_shape, right_shape):
    # assert l == r or l == 1 or r == 1, f"Cannot broadcast shapes {left.shape}, {right.shape}"
    target_shape.append(BinaryOp(Ops.MAX, l, r).simplify())
  target_shape = tuple(target_shape)
  if left_shape != target_shape: left = Expand(left, target_shape)
  if right_shape != target_shape: right = Expand(right, target_shape)
  return left, right

def ensure_return(node:Node) -> Node:
  # TODO assert all code paths return
  if isinstance(node, Return): return node
  if isinstance(node, Block): return ensure_return(node.nodes[-1])
  return Return(node)

def assert_unique_dims(dims:tuple[int], shape:tuple):
  assert len(dims) > 0, f"Dims cannot be empty"
  assert len(dims) == len(set(dims)), f"Dims {dims} need to be unique"
  assert len(dims) <= len(shape), f"Too many dims for shape {shape}. Dims: {dims}"
  assert all(isinstance(d, int) for d in dims), f"Dims need to be all ints. Dims {dims}"