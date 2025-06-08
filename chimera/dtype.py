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
  int32 = DType(4, 'int', 'i')
  float32 = DType(4, 'float', 'f')
  void = DType(0, 'void', None)

  python_to_dtype = {int:int32, float:float32}