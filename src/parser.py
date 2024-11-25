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
    CONST = auto()
    ASSIGN = auto()
    PRINT = auto()
    DEFINE = auto()
    DATA = auto()
    COLON = auto()
    ARROW = auto()
    DOT = auto()
    PARENBEGIN = auto() # can maybe just be id
    PARENEND = auto()
    ID = auto()
    NEWLINE = auto()
    SKIP = auto()
    MISMATCH = auto()

class Ops(FastEnum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    CONST = auto()
    GROUP = auto()
    PRINT = auto()

class OpGroup:
    Binary = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.DIV}
    Terminal = {Ops.PRINT}

class TypeGroup:
    Number = {int, float}

token_specification = list(map(lambda t: (t[0], re.compile(t[1])), [
    (Token.CONST, r"\d+(\.\d*)?"),  # Integer or decimal number
    (Token.ASSIGN, r"="),  # Assignment operator
    (Token.PRINT, r"print "),  # Print
    (Token.DEFINE, r"define "),  # Define
    (Token.DATA, r"data "),  # Data
    (Token.COLON, r":"),  # Colon
    (Token.ARROW, r"->"),  # Return type arrow
    (Token.DOT, r"\."),  # .
    (Token.PARENBEGIN, r"\("),  # (
    (Token.PARENEND, r"\)"),  # )
    (Token.ID, r"[A-Za-z_+\-*/]\w*"),  # Identifiers
    (Token.NEWLINE, r"\n"),  # Line endings
    (Token.SKIP, r"[ \t]+"),  # Skip over spaces and tabs
    (Token.MISMATCH, r"."),  # Any other character
]))

class Symbol():
    __slots__ = ["id", "value", "dtype", "sources"]
    def __init__(self, id:FastEnum, value:any=None, dtype:type=None, sources=tuple()):
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
    def __init__(self, patterns:list[Tuple[Union[Pattern, Tuple[Pattern]], Callable]]):
        self.patterns:list[Tuple[Tuple[Pattern], Callable]] = [(pat if isinstance(pat, tuple) else (pat,), fxn) for pat,fxn in patterns]
        
    def rewrite(self, symbols:List[Symbol], ctx=None) -> Tuple[bool, int, any]:
        for pattern_tuple, fxn in self.patterns:
            pattern_index = 0 
            pattern_length = 0
            args: Dict[str, Union[Symbol, Tuple[Symbol]]] = {}
            for contender in symbols:
                if pattern_index >= len(pattern_tuple): break
                matcher = pattern_tuple[pattern_index]
                if isinstance(matcher, ElipticalPattern): # TODO Make clean
                    success = pattern_tuple[pattern_index + 1].match(contender, args)
                    if success:
                        pattern_index += 2
                        if matcher.name is not None: args[matcher.name] = tuple(symbols[pattern_index:pattern_length])
                    pattern_length += 1
                    continue
                success = matcher.match(contender, args)
                if not success: break
                pattern_index += 1
                pattern_length += 1
            if pattern_index >= len(pattern_tuple):
                return (True, pattern_length, fxn(ctx=ctx, **args))
        return (False, 0, None)

def tokenize(code: str) -> List[Tuple[Token, Any]]:
    position = 0
    symbols = []
    while position < len(code):
        match = None
        for token, pattern_tuple in token_specification:
            match = pattern_tuple.match(code, position)
            if not match: continue
            value = match.group()
            position = match.end()
            if token is Token.SKIP: break
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

class Parser:
    precedence: Dict[int, List[str]] = {2: ['+', '-'], 3: ['*', '/'], 4: ['(', ')'] }
    all_patterns:List[Tuple[Tuple[Pattern], Callable]] = [
        ((Pattern(Token.PARENBEGIN, value='('), ElipticalPattern(name='x'), Pattern(Token.PARENEND, value=')')), lambda ctx,x: Symbol(Ops.GROUP, sources=x)),
        ((Pattern(name='a', dtype=TypeGroup.Number), Pattern(Token.ID, '+'), Pattern(name='b', dtype=TypeGroup.Number)), lambda ctx,a,b: Symbol(Ops.ADD, dtype=resolve_number((a, b)), sources=(a,b))),
        ((Pattern(name='a', dtype=TypeGroup.Number), Pattern(Token.ID, '-'), Pattern(name='b', dtype=TypeGroup.Number)), lambda ctx,a,b: Symbol(Ops.SUB, dtype=resolve_number((a, b)), sources=(a,b))),
        ((Pattern(name='a', dtype=TypeGroup.Number), Pattern(Token.ID, '*'), Pattern(name='b', dtype=TypeGroup.Number)), lambda ctx,a,b: Symbol(Ops.MUL, dtype=resolve_number((a, b)), sources=(a,b))),
        ((Pattern(name='a', dtype=TypeGroup.Number), Pattern(Token.ID, '/'), Pattern(name='b', dtype=TypeGroup.Number)), lambda ctx,a,b: get_div(a,b)),
        ((Pattern(Token.CONST, name='x'),), lambda ctx,x: Symbol(Ops.CONST, x.value, x.dtype)),
        ((Pattern(Token.PRINT), Pattern(name='x')), lambda ctx,x: Symbol(Ops.PRINT, sources=x)),
        ((Pattern(Token.NEWLINE),), lambda ctx: None),
    ]

    def __init__(self) -> None: 
        precedence = dict(sorted(self.precedence.items(), reverse=True))
        lowest_precedence = list(self.all_patterns)
        self.precedence_patterns: Dict[int, PatternMatcher] = {}
        for precedence_level, prec in precedence.items():
            patterns = [(pats,fxn) for pats,fxn in self.all_patterns if any(pat.value in prec for pat in pats)]
            if len(patterns) == 0: continue
            for pattern_tuple in patterns:
                lowest_precedence.remove(pattern_tuple)
            self.precedence_patterns[precedence_level] = PatternMatcher(patterns)
        self.min_precedence_level = list(precedence.keys())[-1]-1
        self.precedence_patterns[self.min_precedence_level] = PatternMatcher(lowest_precedence)

    def precedence_pass(self, symbols: List[Symbol], matcher:PatternMatcher, precedence_level:int):
        for i in reversed(range(len(symbols))):
            success, length, rewrite = matcher.rewrite(symbols[i:])
            if not success: continue
            if rewrite is None:
                del symbols[i:i+length]
                continue
            used_symbols = symbols[i:i+length]
            sources = self.parse_ast(used_symbols, precedence_level-1)
            rewrite.sources = tuple([contender for contender in sources if isinstance(contender.id, Ops)])
            del symbols[i:i+length]
            symbols.insert(i, rewrite)
                

    def parse_ast(self, tokens: List[Symbol], max_precedence_level=100) -> List[Symbol]:
        if self.min_precedence_level > max_precedence_level: return tokens
        ast = list(tokens)
        for precedence_level, matcher in self.precedence_patterns.items():
            if precedence_level > max_precedence_level: continue
            self.precedence_pass(ast, matcher, precedence_level)
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

render_patterns = PatternMatcher([
    (Pattern(Ops.CONST, name='x', dtype=int), lambda ctx, x: f'{x.value}'),
    (Pattern(Ops.CONST, name='x', dtype=float), lambda ctx, x: f'{x.value}f'),
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