import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chimera.parser import tokenize, Parser, linearize, render
from chimera.compiler import compile

code = """
a = 1 + 34 + 2 + 3 * 4 * 5 * 9
print a
"""

def run(code:str):
    tokens = tokenize(code)
    ast = Parser().parse_ast(tokens)
    procedure = linearize(ast)
    rendering = render(procedure)
    return compile(rendering)

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
    print("===LINEARIZE===")
    procedure = linearize(ast)
    for line in procedure:
        print(line)
    print()
    print("===RENDER===")
    rend = render(procedure)
    print(rend)
    print()
    print("===COMPILE===")
    compile(rend)

if __name__ == "__main__":
    main()