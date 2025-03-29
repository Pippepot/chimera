import subprocess, os, time

def compile(code, functions):
  # with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib.c")) as file: lib = file.read()
  code = f'#include <stdio.h>\n\n{functions}\n\nint main(int argc, char *argv[]) {{\n{code}\n}}'
  print(code)
  clang_timer = time.perf_counter()
  subprocess.run(['clang', '-O2', '-Wall', '-Werror', '-x', 'c', '-', '-o', "program.exe"], input=code.encode('utf-8'))
  print(f"Clang compile\t{time.perf_counter() - clang_timer:.4f}ms")
  runtime_timer = time.perf_counter()
  result = subprocess.run(['./program.exe'], capture_output=True, text=True)
  print(f"Ran in\t\t{time.perf_counter() - runtime_timer:.4f}ms")
  return result.stdout