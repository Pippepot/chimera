from parser import tokenize, Parser, linearize, render
from compiler import compile

code = """
print 2* 2- 1
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