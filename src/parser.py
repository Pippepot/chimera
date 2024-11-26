from __future__ import annotations
from enum import auto, IntEnum, Enum
from typing import Tuple, List, Any, Union, Dict, Callable, Optional, Set
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
    GROUP = auto(); PRINT = auto()

class OpGroup:
    Binary = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.DIV}
    Terminal = {Ops.PRINT, Ops.ASSIGN}

class TypeGroup:
    Number = {int, float}

token_specification = [(token, re.compile(regex)) for token, regex in [
    (Token.CONST, r"\d+(\.\d*)?"),  # Integer or decimal number
    (Token.ALPHABETIC, r"[A-Za-z_]\w*"),  # Identifiers
    (Token.NEWLINE, r"\n"),  # Line endings
    (Token.INDENT, r"[ \t]+"),  # Tabs and spaces
    (Token.SPECIAL, r"."),  # Any other character
]]

class Symbol():
    __slots__ = ["id", "value", "dtype", "sources"]
    def __init__(self, id:FastEnum, value:any=None, dtype:type=None, sources=()):
        self.id = id
        self.value = value
        self.dtype = dtype
        self.sources = sources
    def __repr__(self): return f"Symbol: {self.id}, {repr(self.value)}{'' if self.dtype is None else f', dtype={self.dtype.__name__}'}, src={self.sources}"

class Pattern():
    __slots__ = ["id", "value", "dtype", "sources", "name"]
    def __init__(self, id:Optional[Union[FastEnum, Tuple[FastEnum], Set[FastEnum]]]=None, value:Optional[any]=None,
                 dtype:Optional[Union[type, Tuple[type], Set[type]]]=None, sources:Optional[Union[Pattern, Tuple[Pattern]]]=None, name:Optional[str]=None):
        self.id: Optional[Tuple[FastEnum]] = (id,) if isinstance(id, FastEnum) else tuple(id) if isinstance(id, list) or isinstance(id, set) else id
        self.value = value
        self.dtype: Optional[tuple[type]] = dtype if isinstance(dtype, tuple) or dtype is None else tuple(dtype) if isinstance(dtype, set) else (dtype,)
        self.sources:Optional[Tuple[Pattern]] = sources if isinstance(sources, tuple) or sources is None else (sources,)
        self.name = name
        
    def match(self, symbol:Symbol, args:Dict[str, Union[Symbol, Tuple[Symbol]]]) -> bool:
        if (self.id is not None and symbol.id not in self.id) or \
           (self.value is not None and symbol.value != self.value) or \
           (self.dtype is not None and symbol.dtype not in self.dtype): return False
        if self.sources is not None:
            if len(self.sources) != len(symbol.sources): return False
            if not all([pattern.match(source, args) for pattern, source in zip(self.sources, symbol.sources)]): return False
        if self.name is not None: args[self.name] = symbol
        return True
    
    def __repr__(self):
        return f"Pat({self.id}, {self.value}, {self.name})"

class ElipticalPattern(Pattern):
    pass

class PatternMatcher:
    def __init__(self, patterns_list:list[Tuple[Union[Pattern, Tuple[Pattern]], Callable]]):
        self.patterns_list:list[Tuple[Tuple[Pattern], Callable]] = [(pat if isinstance(pat, tuple) else (pat,), fxn) for pat,fxn in patterns_list]
        
    def rewrite(self, symbols:List[Symbol], ctx=None) -> Tuple[bool, int, any]:
        for patterns, fxn in self.patterns_list:
            pattern_index = 0 
            pattern_length = 0
            args: Dict[str, Union[Symbol, Tuple[Symbol]]] = {}
            for contender in symbols:
                if pattern_index >= len(patterns): break
                pattern = patterns[pattern_index]
                if isinstance(pattern, ElipticalPattern): # TODO Make clean
                    success = patterns[pattern_index + 1].match(contender, args)
                    if success:
                        pattern_index += 2
                        if pattern.name is not None: args[pattern.name] = tuple(symbols[pattern_index:pattern_length])
                    pattern_length += 1
                    continue
                success = pattern.match(contender, args)
                if not success: break
                pattern_index += 1
                pattern_length += 1
            if pattern_index >= len(patterns):
                return (True, pattern_length, fxn(ctx=ctx, **args))
        return (False, 0, None)

def tokenize(code: str) -> List[Tuple[Token, Any]]:
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

def resolve_number(symbols:Tuple[Symbol]) -> type:
    return float if float in [symbol.dtype for symbol in symbols] else int

def get_div(a:Symbol, b:Symbol):
    if a.dtype is not float and b.dtype is not float:
        if type(b.value) is int: b.value = float(b.value)
        b.dtype = float
    return Symbol(Ops.DIV, dtype=float, sources=(a,b))

def assign_variable(variable:Symbol, value:Symbol, ctx:Dict[str, Symbol]) -> Symbol:
    symbol = Symbol(Ops.ASSIGN, variable.value, value.dtype, (value,))
    ctx[variable.value] = symbol
    return symbol

class Parser:
    precedence: Dict[int, List[str]] = {2: ['+', '-'], 3: ['*', '/'], 4: ['(', ')'], 5: ['=']}
    all_patterns:List[Tuple[Tuple[Pattern], Callable]] = [
        ((Pattern(value='('), ElipticalPattern(name='x'), Pattern(value=')')), lambda ctx,x: Symbol(Ops.GROUP, sources=x)),
        ((Pattern(name='a', dtype=TypeGroup.Number), Pattern(value='+'), Pattern(name='b', dtype=TypeGroup.Number)), lambda ctx,a,b: Symbol(Ops.ADD, dtype=resolve_number((a, b)), sources=(a,b))),
        ((Pattern(name='a', dtype=TypeGroup.Number), Pattern(value='-'), Pattern(name='b', dtype=TypeGroup.Number)), lambda ctx,a,b: Symbol(Ops.SUB, dtype=resolve_number((a, b)), sources=(a,b))),
        ((Pattern(name='a', dtype=TypeGroup.Number), Pattern(value='*'), Pattern(name='b', dtype=TypeGroup.Number)), lambda ctx,a,b: Symbol(Ops.MUL, dtype=resolve_number((a, b)), sources=(a,b))),
        ((Pattern(name='a', dtype=TypeGroup.Number), Pattern(value='/'), Pattern(name='b', dtype=TypeGroup.Number)), lambda ctx,a,b: get_div(a,b)),
        ((Pattern(Token.CONST, name='x'),), lambda ctx,x: Symbol(Ops.CONST, x.value, x.dtype)),
        ((Pattern(value='print'), Pattern(name='x')), lambda ctx,x: Symbol(Ops.PRINT, sources=x)),
        ((Pattern(Token.ALPHABETIC, name='a'), Pattern(value='='), Pattern(name='b')), lambda ctx,a,b: assign_variable(a, b, ctx)),
        ((Pattern(Token.ALPHABETIC, name='x'),), lambda ctx,x: Symbol(Ops.LOAD, x.value, ctx[x.value].dtype, sources=(ctx[x.value],))),
        ((Pattern(Token.NEWLINE),), lambda ctx: None),
    ]

    def __init__(self) -> None: 
        precedence = dict(sorted(self.precedence.items(), reverse=True))
        lowest_precedence = list(self.all_patterns)
        self.precedence_patterns: Dict[int, PatternMatcher] = {}
        for precedence_level, prec in precedence.items():
            patterns_list = [(pats,fxn) for pats,fxn in self.all_patterns if any(pat.value in prec for pat in pats)]
            if len(patterns_list) == 0: continue
            for patterns in patterns_list:
                lowest_precedence.remove(patterns)
            self.precedence_patterns[precedence_level] = PatternMatcher(patterns_list)
        self.min_precedence_level = list(precedence.keys())[-1]-1
        self.precedence_patterns[self.min_precedence_level] = PatternMatcher(lowest_precedence)

    def precedence_pass(self, symbols: List[Symbol], pattern:PatternMatcher, variables:Dict[str, Symbol], precedence_level:int):
        for i in reversed(range(len(symbols))):
            success, length, rewrite = pattern.rewrite(symbols[i:], variables)
            if not success: continue
            if rewrite is None:
                del symbols[i:i+length]
                continue
            used_symbols = symbols[i:i+length]
            sources = self.parse_ast(used_symbols, variables, precedence_level-1)
            rewrite.sources = tuple([contender for contender in sources if isinstance(contender.id, Ops)])
            del symbols[i:i+length]
            symbols.insert(i, rewrite)
                

    def parse_ast(self, tokens: List[Symbol], variables:Dict[str, Symbol]={}, max_precedence_level=100) -> List[Symbol]:
        if self.min_precedence_level > max_precedence_level: return tokens
        ast = list(tokens)
        for precedence_level, pattern in self.precedence_patterns.items():
            if precedence_level > max_precedence_level: continue
            self.precedence_pass(ast, pattern, variables, precedence_level)
        return ast

def get_children_dfs(sym:Symbol, children:Dict[Symbol, List[Symbol]]):
    if sym in children: return
    children[sym] = []
    for source in sym.sources:
        get_children_dfs(source, children)
        children[source].append(sym)

def linearize(ast:List[Symbol]) -> List[Symbol]:
    children:Dict[Symbol, List[Symbol]] = {}
    for node in ast: get_children_dfs(node, children)
    return list(reversed(children.keys()))

op_patterns: Dict = {
    Ops.ADD: lambda a,b: f"({a}+{b})",
    Ops.SUB: lambda a,b: f"({a}-{b})",
    Ops.MUL: lambda a,b: f"({a}*{b})",
    Ops.DIV: lambda a,b: f"({a}/{b})",
}

type_map = {int:'int', float:'float'}
render_patterns = PatternMatcher([
    (Pattern(Ops.CONST, name='x', dtype=int), lambda ctx, x: f'{x.value}'),
    (Pattern(Ops.CONST, name='x', dtype=float), lambda ctx, x: f'{x.value}f'),
    (Pattern(Ops.ASSIGN, name='x'), lambda ctx, x: f'{type_map[x.dtype]} {x.value} = {ctx[x.sources[1]]};'),
    (Pattern(Ops.LOAD, name='x'), lambda ctx, x: f'{x.value}'),
    (Pattern(Ops.PRINT, sources=Pattern(name='x', dtype=int)), lambda ctx, x: r'printf("%d\n",' + f'{ctx[x]});'),
    (Pattern(Ops.PRINT, sources=Pattern(name='x', dtype=float)), lambda ctx, x: r'printf("%f\n",' + f'{ctx[x]});'),
    (Pattern(OpGroup.Binary, name='x', dtype=TypeGroup.Number), lambda ctx, x: op_patterns[x.id](
        *[ctx[source] for source in x.sources]
    )),
])

def render(symbols: List[Symbol]) -> str:
    refs: Dict[Symbol, str] = {}
    body = []
    for contender in symbols:
        success, _, value = render_patterns.rewrite((contender,), ctx=refs)
        if not success: continue
        if contender.id in {Ops.CONST, *OpGroup.Binary}:
            refs[contender] = value
        elif contender.id in OpGroup.Terminal:
            body.append(value)
    return '\n'.join(body)