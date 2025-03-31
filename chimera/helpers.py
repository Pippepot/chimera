from typing import TypeVar, Iterable
import functools, operator
T = TypeVar("T")

def prod(x:Iterable[T]) -> T|int: return functools.reduce(operator.mul, x, 1)
def tupled(x) -> tuple: return tuple(x) if isinstance(x, Iterable) else (x,)
def listed(x) -> list: return list(x) if isinstance(x, Iterable) else [x]
def all_same(items:tuple[T, ...]|list[T]): return all(x == items[0] for x in items)
def all_instance(items:Iterable[T], types:tuple[type]|type): return all(isinstance(x, types) for x in items)
def get_shape(x) -> tuple[int, ...]:
  if not hasattr(x, "__len__") or not hasattr(x, "__getitem__") or isinstance(x, str): return ()
  if not all_same(subs:=[get_shape(xi) for xi in x]): raise ValueError(f"inhomogeneous shape from {x}")
  return (len(subs),) + (subs[0] if subs else ())
def fully_flatten(l):
  if hasattr(l, "__len__") and hasattr(l, "__getitem__") and not isinstance(l, str):
    flattened = []
    for li in l: flattened.extend(fully_flatten(li))
    return flattened
  return [l]