import subprocess, os, time
from chimera.helpers import DEBUG, OPTIMIZE

def compile(code, functions):
  with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib.c")) as file: lib = file.read()
  libs = f'{lib}\n\n{functions}\n\n'
  main = f"int main(int argc, char *argv[]) {{\n{code}\n}}"
  if DEBUG:
    print(f"\n{code}")
    clang_timer = time.perf_counter()

  opt = '-O3' if OPTIMIZE else ''
  process = subprocess.run(['clang', opt, '-Wall', '-Werror', '-x', 'c', '-', '-o', "program.exe"], input=f'{libs}{main}'.encode('utf-8'))
  if DEBUG:
    print(f"Clang compile\t{time.perf_counter() - clang_timer:.4f}s")
    runtime_timer = time.perf_counter()

  if process.returncode != 0: return "Compilation failed"

  result = subprocess.run(['./program.exe'], capture_output=True, text=True)
  if DEBUG:
    print(f"Ran in\t\t{(time.perf_counter() - runtime_timer) * 1000:.1f}ms")
  return result.stdout