import subprocess, os, time
from chimera.helpers import DEBUG
import tempfile

def compile(code, functions):
  with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib.c")) as file: lib = file.read()
  libs = f'{lib}\n\n{functions}\n\n'
  main = f"int main() {{\n{code}\n}}"
  if DEBUG:
    print(f"{functions}\n{code}\n")
    clang_timer = time.perf_counter()

  with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as exe_file:
    exe_path = exe_file.name

  try:
    process = subprocess.run(['clang', '-O2', '-Wall', '-Werror', '-x', 'c', '-', '-o', exe_path],input=f'{libs}{main}'.encode('utf-8'))
    if DEBUG:
      print(f"Clang compile\t{(time.perf_counter() - clang_timer) * 1000:.1f}ms")
      runtime_timer = time.perf_counter()

    if process.returncode != 0:
      return "Compilation failed"

    result = subprocess.run([exe_path], capture_output=True, text=True)
    if DEBUG:
      print(f"Ran in\t\t{(time.perf_counter() - runtime_timer) * 1000:.1f}ms")
    return result.stdout
  finally:
    try:
      os.remove(exe_path)
    except Exception:
      pass