from parser import tokenize, Parser, linearize, render
from compiler import compile

code = """
a:int = 3
print 2 * a - 2 / 7.1
print a + 3
"""

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