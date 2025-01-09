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
    NEWLINE = auto(); INDENT = auto(); DEDENT = auto()

class Ops(FastEnum):
    ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto()
    CONST = auto(); ASSIGN = auto(); LOAD = auto()
    GROUP = auto(); FUNCTION = auto(); PRINT = auto()
    ERROR = auto()

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
    __slots__ = ["id", "value", "dtype", "sources", "row", "column_start", "column_end"]
    def __init__(self, id:FastEnum, value:any=None, dtype:type=None, sources=(), row:int=0, column_start:int=0, column_end:int=0):
        self.id = id
        self.value = value
        self.dtype = dtype
        self.row, self.column_start, self.column_end = row, column_start, column_end
        self.sources = sources if isinstance(sources, tuple) else (sources,)
    def __repr__(self): return f"Symbol: {self.id}, {repr(self.value)}{'' if self.dtype is None else f', dtype={self.dtype.__name__}'}, src={self.sources}"

class Function(Symbol):
    __slots__ = ["block", "args"]
    def __init__(self, block:list[Symbol], args:tuple[Symbol], returnType:str=None):
        self.block, self.args = block, args
        dtype = types[returnType] if returnType is not None else None
        # Maybe find terminal lines and use them as sources instead of having a seperate block field
        super().__init__(Ops.FUNCTION, None, dtype, ())

class Pat():
    __slots__ = ["id", "value", "dtype", "sources", "name"]
    def __init__(self, id:FastEnum|tuple[FastEnum]|set[FastEnum]=None, value:any|tuple[any]=None,
                 dtype:type|tuple[type]|set[type]=None, sources:Pat|tuple[Pat]=None, name:str=None):
        self.id: tuple[FastEnum] = id if isinstance(id, tuple) or id is None else tuple(id) if isinstance(id, list) or isinstance(id, set) else (id,)
        self.value: tuple[any] = value if isinstance(value, tuple) or value is None else (value,)
        self.dtype: tuple[type] = dtype if isinstance(dtype, tuple) or dtype is None else tuple(dtype) if isinstance(dtype, set) else (dtype,)
        self.sources: tuple[Pat] = sources if isinstance(sources, tuple) or sources is None else (sources,)
        self.name = name
        
    def match(self, symbols:list[Symbol], index:int, args:dict[str, Symbol|tuple[Symbol]]) -> int:
        symbol = symbols[index]
        if (self.id is not None and symbol.id not in self.id) or \
           (self.value is not None and symbol.value not in self.value) or \
           (self.dtype is not None and symbol.dtype not in self.dtype): return None
        if self.sources is not None:
            if len(self.sources) != len(symbol.sources): return None
            if not all([pattern.match([source], 0, args) is not None for pattern, source in zip(self.sources, symbol.sources)]): return None
        if self.name is not None: args[self.name] = symbol
        return 1
    
    def __repr__(self):
        return f"Pat({self.id}, {self.value}, {self.name})"

class ScopePat():
    __slots__ = ["name"]
    def __init__(self, name:str=None):
        self.name=name
    
    def match(self, symbols:list[Symbol], index:int, args:dict[str, Symbol|tuple[Symbol]]):
        scope_is_single_line = symbols[index].id is not Token.NEWLINE
        end_index = index
        indentation = 0
        while end_index < len(symbols):
            symbol = symbols[end_index]
            if scope_is_single_line and symbol.id is Token.NEWLINE:
                break
            end_index += 1
            if symbol.id is Token.INDENT: indentation += 1
            if symbol.id is Token.DEDENT: indentation -= 1
            if indentation == 0: break

        consumed = symbols[index:end_index]
        args[self.name] = tuple(consumed)
        return len(consumed)


class ElipticalPat():
    __slots__ = ["name"]
    def __init__(self, name:str=None):
        self.name=name

class PatternMatcher:
    def __init__(self, patterns_list:list[tuple[Pat|tuple[Pat], callable]]):
        self.patterns_list:list[tuple[tuple[Pat], callable]] = [(pat if isinstance(pat, tuple) else (pat,), fxn) for pat,fxn in patterns_list]
        
    def rewrite(self, symbols:list[Symbol], ctx=None) -> tuple[bool, int, any]:
        for patterns, fxn in self.patterns_list:
            pattern_index = 0 
            symbol_index = 0
            args: dict[str, Symbol|tuple[Symbol]] = {}
            while symbol_index < len(symbols) and pattern_index < len(patterns):
                pattern = patterns[pattern_index]

                if isinstance(pattern, ElipticalPat): # TODO Make clean
                    success = patterns[pattern_index + 1].match(symbols, symbol_index, args)
                    if success:
                        if pattern.name is not None: args[pattern.name] = tuple(symbols[pattern_index:symbol_index]) # Does this indexing always work?
                        pattern_index += 2
                    symbol_index += 1
                    continue

                consumed = pattern.match(symbols, symbol_index, args)
                if consumed == None: break
                symbol_index += consumed
                pattern_index += 1
            if pattern_index >= len(patterns):
                return (True, symbol_index, fxn(ctx=ctx, **args))
        return (False, 0, None)

def tokenize(code: str) -> list[tuple[Token, any]]:
    position, row, column, indentation, last_indentation = 0,0,0,0,0
    symbols = []
    while position < len(code):
        match = None
        for token, regex in token_specification:
            match = regex.match(code, position)
            if not match: continue
            value = match.group()
            position = match.end()
            column_start = column
            column += len(value)
            if token is Token.INDENT:
                if not symbols or symbols[-1].id is Token.NEWLINE: indentation += 1
                break
            # Append INDENT or DEDENT when the indentation is differs from last line
            symbols.extend([Symbol(Token.INDENT if indentation > last_indentation else Token.DEDENT) for _ in range(abs(last_indentation - indentation))])
            last_indentation = indentation
            if token is Token.NEWLINE:
                indentation, column, column_start = 0, 0, 0
                row += 1
            if token is Token.CONST: value = float(value) if "." in value else int(value)
            symbols.append(Symbol(token, value, type(value), row=row, column_start=column_start, column_end=column))
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

def load_variable(ctx:dict[str, Symbol], variable:Symbol):
    if variable.value not in ctx: return Symbol(Ops.ERROR, f"Undeclared identifier '{variable.value}'")
    return Symbol(Ops.LOAD, variable.value, ctx[variable.value].dtype, sources=(ctx[variable.value],))

class Parser:
    precedence: dict[int, list[str]] = {2: ['+', '-'], 3: ['*', '/'], 4:['\\'], 5: ['(', ')']}
    all_patterns:list[tuple[tuple[Pat], callable]] = [
        ((Pat(value='('), ElipticalPat(name='x'), Pat(value=')')), lambda ctx,x: Symbol(Ops.GROUP, sources=x)),
        ((Pat(name='a'), Pat(value='+'), Pat(name='b')), lambda ctx, a, b: Symbol(Ops.ADD, dtype=resolve_number((a, b)), sources=(a, b))),
        ((Pat(name='a'), Pat(value='-'), Pat(name='b')), lambda ctx, a, b: Symbol(Ops.SUB, dtype=resolve_number((a, b)), sources=(a, b))),
        ((Pat(name='a'), Pat(value='*'), Pat(name='b')), lambda ctx, a, b: Symbol(Ops.MUL, dtype=resolve_number((a, b)), sources=(a, b))),
        ((Pat(name='a'), Pat(value='/'), Pat(name='b')), lambda ctx, a, b: get_div(a,b)),
        ((Pat(Token.CONST, name='x'),), lambda ctx, x: Symbol(Ops.CONST, x.value, x.dtype)),
        ((Pat(value='\\'), ElipticalPat(name='args'), Pat(value='->'), Pat(name='ret', value=tuple(types.keys())), ScopePat(name="block")),
         lambda ctx, args, ret, block: Function(block, args, ret)),
        ((Pat(value='\\'), ElipticalPat(name='args'), Pat(value='->'), ScopePat(name="block")), lambda ctx, args, block: Function(block, args, None)),
        ((Pat(value='print'), Pat(name='x')), lambda ctx, x: Symbol(Ops.PRINT, sources=x)),
        ((Pat(Token.ALPHABETIC, name='a'), Pat(value='='), Pat(name='b')), lambda ctx, a, b: assign_variable(a, b, ctx)),
        ((Pat(Token.ALPHABETIC, name='x'),), lambda ctx, x: load_variable(ctx, x)),
    ]

    def __init__(self) -> None: 
        precedence = dict(sorted(self.precedence.items(), reverse=True))
        lowest_precedence = list(self.all_patterns)
        self.precedence_patterns: dict[int, PatternMatcher] = {}
        for precedence_level, prec in precedence.items():
            patterns_list = [(pats,fxn) for pats,fxn in self.all_patterns if any(pat.value is not None and bool(set(pat.value) & set(prec)) for pat in pats if isinstance(pat, Pat))]
            if len(patterns_list) == 0: continue
            for patterns in patterns_list:
                lowest_precedence.remove(patterns)
            self.precedence_patterns[precedence_level] = PatternMatcher(patterns_list)
        self.min_precedence_level = min(precedence.keys())-1
        self.precedence_patterns[self.min_precedence_level] = PatternMatcher(lowest_precedence)


    def precedence_pass(self, symbols: list[Symbol], pattern:PatternMatcher, variables:dict[str, Symbol], precedence_level:int):
        i = 0
        while i < len(symbols):
            rewrite:Symbol
            success, length, rewrite = pattern.rewrite(symbols[i:], variables)
            if not success:
                i += 1
                continue

            if rewrite.column_end is 0: rewrite.row, rewrite.column_start, rewrite.column_end = symbols[i].row, symbols[i].column_start, symbols[i+length-1].column_end
            del symbols[i:i+length]
            if rewrite.sources:
                rewrite.sources = tuple(self.parse_line(list(rewrite.sources), variables, precedence_level))
                if rewrite.dtype is None and len(rewrite.sources) == 1: rewrite.dtype = rewrite.sources[0].dtype
            symbols.insert(i, rewrite)
                

    # TODO We should not parse single lines as functions span multiple lines
    def parse_line(self, tokens: list[Symbol], variables:dict[str, Symbol], max_precedence_level:int) -> list[Symbol]:
        if self.min_precedence_level > max_precedence_level: return tokens
        line = tokens.copy() # This can technically be removed since parse_ast does not use lines after parsing. Copy anyway to avoid nasty bugs in the future
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
    (Pat(Ops.FUNCTION, name='x'), lambda ctx, x: f'{type_to_str[x.dtype]} anon(){{}}'),
    (Pat(Ops.PRINT, sources=Pat(name='x', dtype=int)), lambda ctx, x: r'printf("%d\n",' + f'{ctx[x]});'),
    (Pat(Ops.PRINT, sources=Pat(name='x', dtype=float)), lambda ctx, x: r'printf("%f\n",' + f'{ctx[x]});'),
    (Pat(OpGroup.Binary, name='x', dtype=TypeGroup.Number), lambda ctx, x: op_patterns[x.id](
        *[ctx[source][1:-1] if x.id == source.id and x.id in OpGroup.Associative else ctx[source] for source in x.sources]
    )),
])

def render(symbols: list[Symbol]) -> tuple[str, str]:
    refs: dict[Symbol, str] = {}
    functions = []

    for i, func in enumerate([s for s in symbols if s.id is Ops.FUNCTION]):
        name = f"func_{i}_{type_to_str[func.dtype]}()"
        refs[func] = name
        functions.append(f"{type_to_str[func.dtype]} {name} {{ return 0; }}")

    body = []
    indent = 1
    for contender in symbols:
        if contender.id is Ops.FUNCTION: continue
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