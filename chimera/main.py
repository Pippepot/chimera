import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chimera.parser import tokenize, Parser, linearize, render
from chimera.compiler import compile

code = """
foo = \\n -> int
    
print (foo)
"""

def run(code:str):
    tokens = tokenize(code)
    ast = Parser().parse_ast(tokens)
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

if __name__ == "__main__":
    main()