import ctypes, subprocess, tempfile
import time
from parser import Symbol

def compile(code: str):
    code = f'#include <stdio.h>\nint main(int argc, char *argv[]) {{\n {code} \n}}'
    print(code)
    subprocess.run(['clang', '-O2', '-Wall', '-Werror', '-x', 'c', '-', '-o', "HELLO.exe"], input=code.encode('utf-8'))
    st = time.perf_counter()
    result = subprocess.run(['./HELLO.exe', "beter"], capture_output=True, text=True)
    t = time.perf_counter()-st
    print(f"Ran in {t}ms")
    print(result.stdout)
 