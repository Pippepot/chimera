import ctypes, subprocess, tempfile
import time
from chimera.parser import Symbol

def compile(code: str, functions:str):
    code = f'#include <stdio.h>\n\n{functions}\n\nint main(int argc, char *argv[]) {{\n{code}\n}}'
    print(code)
    subprocess.run(['clang', '-O2', '-Wall', '-Werror', '-x', 'c', '-', '-o', "program.exe"], input=code.encode('utf-8'))
    st = time.perf_counter()
    result = subprocess.run(['./program.exe'], capture_output=True, text=True)
    t = time.perf_counter()-st
    print(f"Ran in {t}ms")
    print(result.stdout)
    return result.stdout
 