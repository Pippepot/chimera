from typing import TypeVar, Iterable
import sys, os
import functools, operator
T = TypeVar("T")

ARGS = {k.upper(): v for k, v in (arg.split('=') for arg in sys.argv[1:] if '=' in arg)}

class CompileOption:
    value: int
    key: str
    def __init__(self, key:str, default_value:int=0):
        self.key = key.upper()
        self.value = ARGS.get(self.key, os.getenv(self.key, default_value))
        try: self.value = int(self.value)
        except ValueError:
            raise ValueError(f"Invalid value for {self.key}: {self.value}. Expected an integer.")
    def __bool__(self): return bool(self.value)
    def __ge__(self, x): return self.value >= x
    def __gt__(self, x): return self.value > x
    def __lt__(self, x): return self.value < x

DEBUG, OPTIMIZE = CompileOption("DEBUG"), CompileOption("OPTIMIZE", 1)

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