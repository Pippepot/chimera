from dataclasses import dataclass
from typing import Literal, Optional

FmtStr = Literal['?', 'b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q', 'e', 'f', 'd']

@dataclass(frozen=True, eq=False)
class DType:
  itemsize: int
  name: str
  fmt: Optional[FmtStr]

  def new(itemsize:int, name:str, fmt:Optional[FmtStr]): return DType(itemsize, name, fmt)

class dtypes:
  int = DType(4, 'int', 'i')
  float = DType(4, 'float', 'f')
  bool = DType(1, 'bool', 'b')
  void = DType(0, 'void', None)

  def get_dtype(value) -> DType: return getattr(dtypes, type(value).__name__.lower())