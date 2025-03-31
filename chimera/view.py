from __future__ import annotations
import functools, itertools, operator
from dataclasses import dataclass

@functools.lru_cache(maxsize=None)
def canonicalize_strides(shape:tuple[int, ...], strides:tuple[int, ...]) -> tuple[int, ...]:
  return tuple(0 if s == 1 else st for s, st in zip(shape, strides))

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:tuple[int, ...]) -> tuple[int, ...]:
  if not shape: return ()
  strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))[::-1]
  return canonicalize_strides(shape, strides)

@dataclass(frozen=True)
class View():
  shape:tuple[int, ...]
  strides:tuple[int, ...]
  offset:int

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def create(shape:tuple[int, ...]=()) -> View:
    shape = shape
    strides = strides_for_shape(shape)
    offset = 0
    return View(shape, strides, offset)