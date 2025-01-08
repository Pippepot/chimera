from __future__ import annotations
from enum import auto, IntEnum, Enum
from itertools import groupby
import re

# wrapper around IntEnum that preserves Enum.__str__ and makes auto() unique across all FastEnum subclasses
class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  def __repr__(self): return self.__str__()
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])

class Token(FastEnum):
    CONST = auto(); ALPHABETIC = auto(); SPECIAL = auto()
    NEWLINE = auto(); INDENT = auto()

class Ops(FastEnum):
    ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(),
    CONST = auto(); ASSIGN = auto(); LOAD = auto()
    GROUP = auto(); LAMBDA = auto(); PRINT = auto()

class OpGroup:
    Binary = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.DIV}
    Associative = {Ops.ADD, Ops.MUL}
    Terminal = {Ops.PRINT, Ops.ASSIGN}

class TypeGroup:
    Number = {int, float}

types = {'int':int, 'float':float, 'string':str, 'bool':bool}

token_specification = [(token, re.compile(regex)) for token, regex in [
    (Token.CONST, r"\d+(\.\d*)?"),  # Integer or decimal number
    (Token.ALPHABETIC, r"[A-Za-z_]\w*"),  # Identifiers
    (Token.NEWLINE, r"\n"),  # Line endings
    (Token.INDENT, r"[ \t]+"),  # Tabs and spaces
    (Token.SPECIAL, r"->"),  # arrow
    (Token.SPECIAL, r"."),  # any other character
]]

class Symbol():
    __slots__ = ["id", "value", "dtype", "sources"]
    def __init__(self, id:FastEnum, value:any=None, dtype:type=None, sources=()):
        self.id = id
        self.value = value
        self.dtype = dtype
        self.sources = sources if isinstance(sources, tuple) else (sources,)
    def __repr__(self): return f"Symbol: {self.id}, {repr(self.value)}{'' if self.dtype is None else f', dtype={self.dtype.__name__}'}, src={self.sources}"

class Pat():
    __slots__ = ["id", "value", "dtype", "sources", "name"]
    def __init__(self, id:FastEnum|tuple[FastEnum]|set[FastEnum]=None, value:any|tuple[any]=None,
                 dtype:type|tuple[type]|set[type]=None, sources:Pat|tuple[Pat]=None, name:str=None):
        self.id: tuple[FastEnum] = (id,) if isinstance(id, FastEnum) else tuple(id) if isinstance(id, list) or isinstance(id, set) else id
        self.value: tuple[any] = (value,) if not isinstance(value, tuple) and value is not None else value
        self.dtype: tuple[type] = dtype if isinstance(dtype, tuple) or dtype is None else tuple(dtype) if isinstance(dtype, set) else (dtype,)
        self.sources: tuple[Pat] = sources if isinstance(sources, tuple) or sources is None else (sources,)
        self.name = name
        
    def match(self, symbol:Symbol, args:dict[str, Symbol|tuple[Symbol]]) -> bool:
        if (self.id is not None and symbol.id not in self.id) or \
           (self.value is not None and symbol.value not in self.value) or \
           (self.dtype is not None and symbol.dtype not in self.dtype): return False
        if self.sources is not None:
            if len(self.sources) != len(symbol.sources): return False
            if not all([pattern.match(source, args) for pattern, source in zip(self.sources, symbol.sources)]): return False
        if self.name is not None: args[self.name] = symbol
        return True
    
    def __repr__(self):
        return f"Pat({self.id}, {self.value}, {self.name})"

class ElipticalPat(Pat):
    pass

class PatternMatcher:
    def __init__(self, patterns_list:list[tuple[Pat|tuple[Pat], callable]]):
        self.patterns_list:list[tuple[tuple[Pat], callable]] = [(pat if isinstance(pat, tuple) else (pat,), fxn) for pat,fxn in patterns_list]
        
    def rewrite(self, symbols:list[Symbol], ctx=None) -> tuple[bool, int, any]:
        for patterns, fxn in self.patterns_list:
            pattern_index = 0 
            pattern_length = 0
            args: dict[str, Symbol|tuple[Symbol]] = {}
            for contender in symbols:
                if pattern_index >= len(patterns): break
                pattern = patterns[pattern_index]
                if isinstance(pattern, ElipticalPat): # TODO Make clean
                    success = patterns[pattern_index + 1].match(contender, args)
                    if success:
                        if pattern.name is not None: args[pattern.name] = tuple(symbols[pattern_index:pattern_length])
                        pattern_index += 2
                    pattern_length += 1
                    continue
                success = pattern.match(contender, args)
                if not success: break
                pattern_index += 1
                pattern_length += 1
            if pattern_index >= len(patterns):
                return (True, pattern_length, fxn(ctx=ctx, **args))
        return (False, 0, None)

def tokenize(code: str) -> list[tuple[Token, any]]:
    position = 0
    symbols = []
    while position < len(code):
        match = None
        for token, regex in token_specification:
            match = regex.match(code, position)
            if not match: continue
            value = match.group()
            position = match.end()
            if token is Token.INDENT: break
            if token is Token.CONST: value = float(value) if "." in value else int(value)
            symbols.append(Symbol(token, value, type(value)))
            break
    return symbols

def resolve_number(symbols:tuple[Symbol]) -> type:
    return float if float in [symbol.dtype for symbol in symbols] else int

def get_div(a:Symbol, b:Symbol):
    dtype = int if a.dtype is not float and b.dtype is not float else float
    return Symbol(Ops.DIV, dtype=dtype, sources=(a,b))

def assign_variable(variable:Symbol, value:Symbol, ctx:dict[str, Symbol]) -> Symbol:
    symbol = Symbol(Ops.ASSIGN, variable.value, value.dtype, (value,))
    ctx[variable.value] = symbol
    return symbol

# def create_lambda(args:tuple[Symbol], ret:Symbol, ctx:dict[str, Symbol]):
#     definition = Symbol(Ops.DEFINE, name.value, types[ret.value], args)
#     ctx[name.value] = definition
#     return definition

class Parser:
    precedence: dict[int, list[str]] = {2: ['+', '-'], 3: ['*', '/'], 4:['\\'], 5: ['(', ')']}
    all_patterns:list[tuple[tuple[Pat], callable]] = [
        ((Pat(value='('), ElipticalPat(name='x'), Pat(value=')')), lambda ctx,x: Symbol(Ops.GROUP, sources=x)),
        ((Pat(name='a'), Pat(value='+'), Pat(name='b')), lambda ctx,a,b: Symbol(Ops.ADD, dtype=resolve_number((a, b)), sources=(a,b))),
        ((Pat(name='a'), Pat(value='-'), Pat(name='b')), lambda ctx,a,b: Symbol(Ops.SUB, dtype=resolve_number((a, b)), sources=(a,b))),
        ((Pat(name='a'), Pat(value='*'), Pat(name='b')), lambda ctx,a,b: Symbol(Ops.MUL, dtype=resolve_number((a, b)), sources=(a,b))),
        ((Pat(name='a'), Pat(value='/'), Pat(name='b')), lambda ctx,a,b: get_div(a,b)),
        ((Pat(Token.CONST, name='x'),), lambda ctx,x: Symbol(Ops.CONST, x.value, x.dtype)),
        ((Pat(value='\\'), ElipticalPat(name='args'), Pat(value='->'), Pat(name='ret', value=tuple(types.keys()))), lambda ctx,args,ret: Symbol(Ops.LAMBDA, dtype=types[ret.value])),
        ((Pat(value='print'), Pat(name='x')), lambda ctx,x: Symbol(Ops.PRINT, sources=x)),
        ((Pat(Token.ALPHABETIC, name='a'), Pat(value='='), Pat(name='b')), lambda ctx,a,b: assign_variable(a, b, ctx)),
        ((Pat(Token.ALPHABETIC, name='x'),), lambda ctx,x: Symbol(Ops.LOAD, x.value, ctx[x.value].dtype, sources=(ctx[x.value],))),
    ]

    def __init__(self) -> None: 
        precedence = dict(sorted(self.precedence.items(), reverse=True))
        lowest_precedence = list(self.all_patterns)
        self.precedence_patterns: dict[int, PatternMatcher] = {}
        for precedence_level, prec in precedence.items():
            patterns_list = [(pats,fxn) for pats,fxn in self.all_patterns if any(pat.value is not None and bool(set(pat.value) & set(prec)) for pat in pats)]
            if len(patterns_list) == 0: continue
            for patterns in patterns_list:
                lowest_precedence.remove(patterns)
            self.precedence_patterns[precedence_level] = PatternMatcher(patterns_list)
        self.min_precedence_level = min(precedence.keys())-1
        self.precedence_patterns[self.min_precedence_level] = PatternMatcher(lowest_precedence)

    def precedence_pass(self, symbols: list[Symbol], pattern:PatternMatcher, variables:dict[str, Symbol], precedence_level:int):
        i = 0
        while i < len(symbols):
            success, length, rewrite = pattern.rewrite(symbols[i:], variables)
            if not success:
                i += 1
                continue
            
            print(rewrite)
            del symbols[i:i+length]
            if rewrite.sources:
                rewrite.sources = tuple(self.parse_line(list(rewrite.sources), variables, precedence_level))
                if rewrite.dtype is None and len(rewrite.sources) == 1: rewrite.dtype = rewrite.sources[0].dtype
            symbols.insert(i, rewrite)
                

    def parse_line(self, tokens: list[Symbol], variables:dict[str, Symbol], max_precedence_level) -> list[Symbol]:
        if self.min_precedence_level > max_precedence_level: return tokens
        line = tokens.copy() # This can technically be removed since parse_ast does not use lines after parsing. Copy anyway to avoid nasy future bug
        for precedence_level, pattern in self.precedence_patterns.items():
            if precedence_level > max_precedence_level: continue
            self.precedence_pass(line, pattern, variables, precedence_level)
        return line
    
    def parse_ast(self, tokens: list[Symbol]) -> list[Symbol]:
        ast:list[Symbol] = []
        variables:dict[str, Symbol] = {}
        lines = [list(g) for is_new_line, g in groupby(tokens, lambda x: x.id is Token.NEWLINE) if not is_new_line]
        for line in lines:
            ast.extend(self.parse_line(line, variables, max(self.precedence.keys())))
        return ast

def get_children_dfs(sym:Symbol, children:dict[Symbol, list[Symbol]]):
    if sym in children: return
    children[sym] = []
    for source in sym.sources:
        get_children_dfs(source, children)
        children[source].append(sym)

def linearize(ast:list[Symbol]) -> list[Symbol]:
    children:dict[Symbol, list[Symbol]] = {}
    for node in reversed(ast): get_children_dfs(node, children)
    return list(reversed(children.keys()))

op_patterns: dict = {
    Ops.ADD: lambda a,b: f"({a}+{b})",
    Ops.SUB: lambda a,b: f"({a}-{b})",
    Ops.MUL: lambda a,b: f"({a}*{b})",
    Ops.DIV: lambda a,b: f"({a}/{b})",
}

type_to_str = {int:'int', float:'float'}
render_patterns = PatternMatcher([
    (Pat(Ops.CONST, name='x', dtype=int), lambda ctx, x: f'{x.value}'),
    (Pat(Ops.CONST, name='x', dtype=float), lambda ctx, x: f'{x.value}f'),
    (Pat(Ops.ASSIGN, name='x'), lambda ctx, x: f'{type_to_str[x.dtype]} {x.value} = {ctx[x.sources[0]]};'),
    (Pat(Ops.LOAD, name='x'), lambda ctx, x: f'{ctx[x.sources[0]]}'),
    (Pat(Ops.GROUP, name='x'), lambda ctx, x: f'({ctx[x.sources[0]]})'),
    (Pat(Ops.LAMBDA, name='x'), lambda ctx, x: f'{type_to_str[x.dtype]} anon(){{}}'),
    (Pat(Ops.PRINT, sources=Pat(name='x', dtype=int)), lambda ctx, x: r'printf("%d\n",' + f'{ctx[x]});'),
    (Pat(Ops.PRINT, sources=Pat(name='x', dtype=float)), lambda ctx, x: r'printf("%f\n",' + f'{ctx[x]});'),
    (Pat(OpGroup.Binary, name='x', dtype=TypeGroup.Number), lambda ctx, x: op_patterns[x.id](
        *[ctx[source][1:-1] if x.id == source.id and x.id in OpGroup.Associative else ctx[source] for source in x.sources]
    )),
])

def render(symbols: list[Symbol]) -> tuple[str, str]:
    refs: dict[Symbol, str] = {}
    functions = []

    for i, func in enumerate([s for s in symbols if s.id is Ops.LAMBDA]):
        name = f"func_{i}_{type_to_str[func.dtype]}()"
        refs[func] = name
        functions.append(f"{type_to_str[func.dtype]} {name} {{ return 0; }}")

    body = []
    indent = 1
    for contender in symbols:
        if contender.id is Ops.LAMBDA: continue
        success, _, value = render_patterns.rewrite((contender,), ctx=refs)
        if not success:
            print("RENDER: Failed to parse", contender)
            continue

        if contender.id in {Ops.ASSIGN}:
            refs[contender] = contender.value
        if contender.id in {Ops.GROUP, Ops.CONST, Ops.LOAD, *OpGroup.Binary}:
            refs[contender] = value
        elif contender.id in OpGroup.Terminal:
            body.append('  '*indent + value)
    return '\n'.join(body), '\n\n'.join(functions)