import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chimera.parser import tokenize, Symbol, Ops, Parser, linearize, render
from chimera.compiler import compile

code = """
foo = \\n ->
    n
print (foo)
"""

def run(code:str):
    tokens = tokenize(code)
    ast = Parser().parse_ast(tokens)

    errors = [symbol for symbol in ast if symbol.id is Ops.ERROR]
    if errors:
        all_lines = code.splitlines()
        for error in errors:
            print_error(error, all_lines)
        return

    procedure = linearize(ast)
    rendering, functions = render(procedure)
    return compile(rendering, functions)

def main():
    print("===TOKENS===")
    tokens = tokenize(code)
    for t in tokens:
        print(t)
    print()
    print("===AST===")
    ast = Parser().parse_ast(tokens)
    for a in ast:
        print(a)
    print()

    errors = [symbol for symbol in ast if symbol.id is Ops.ERROR]
    if errors:
        all_lines = code.splitlines()
        for error in errors:
            print_error(error, all_lines)
        return
    
    print("===LINEARIZE===")
    procedure = linearize(ast)
    for line in procedure:
        print(line)
        
    print()
    print("===RENDER===")
    rend, functions = render(procedure)
    print(rend)
    print()
    print("===COMPILE===")
    compile(rend, functions)

def print_error(error:Symbol, all_lines:list[str]):
    squggle = " " * error.column_start + "~" * (error.column_end - error.column_start)
    print(f"{error.value} at line {error.row}\n{all_lines[error.row]}\n{squggle}\n")

if __name__ == "__main__":
    main()