from typing import TypeVar, Iterable
import sys, os
import functools, operator, math, shutil
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

DEBUG, TRACK_REWRITES = CompileOption("DEBUG"), CompileOption("TRACK_REWRITES")

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
def navigate_history(get_history_entry, total_entries):
  if get_history_entry is None:
    raise ValueError("get_history_entry must be provided")
  if total_entries < 1:
    raise ValueError("total_entries must be at least 1")
  
  def clear_multiline(text):
    terminal_width = shutil.get_terminal_size().columns
    visual_lines = sum(math.ceil(len(line) / terminal_width) if len(line) > 0 else 1 for line in text.split('\n'))
    for _ in range(visual_lines - 1):
      sys.stdout.write('\r' + ' ' * terminal_width + '\r')
      sys.stdout.write('\x1b[1A')
    sys.stdout.write('\r' + ' ' * terminal_width + '\r')

  def update_text(text, index):
    clear_multiline(text)
    text = f"\n\n ===NAVIGATE HISTORY ({index}/{total_entries-1})===\n\n" + get_history_entry(index)
    print(text, end='', flush=True)
    return text

  current_index = 0
  displayed_text = update_text("", current_index)

  while True:
    key = _get_key()

    if key in (b'\xe0', b'\x00', '\x1b'):
      key = _get_key()
      if key in (b'H', '[A'):
        current_index = max(0, current_index - 1)
      elif key in (b'P', '[B'):
        current_index = min(total_entries - 1, current_index + 1)
      else: continue
    elif key == b'\x1b':
      clear_multiline(displayed_text)
      sys.stdout.flush()
      return
    else: continue
    displayed_text = update_text(displayed_text, current_index)

def _get_key():
  import platform
  if platform.system() == "Windows":
    import msvcrt
    return msvcrt.getch()
  else:
    import tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(3)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch