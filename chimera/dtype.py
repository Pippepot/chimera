class DType:
  def __init__(self, base: str):
    self.base = base
  def __eq__(self, other):
    if not isinstance(other, DType):
      return False
    return self.base == other.base
  def __hash__(self): return hash(self.base)
  def __repr__(self): return self.base

class dtypes:
  int32 = DType("int")
  float32 = DType("float")
  void = DType("void")

  python_to_dtype = {int:int32, float:float32}