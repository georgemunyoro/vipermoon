from dataclasses import dataclass
from parser.scanner.token import Token
from typing import List, Optional, Tuple, Union

from .location import Span


@dataclass
class Node:
    span: Span


@dataclass
class Stat(Node):
    pass


@dataclass
class Block(Node):
    stats: List["Stat"]
    laststat: Optional["LastStat"] = None


@dataclass
class Assignment(Stat):
    targets: List["Var"]
    values: List["Exp"]


@dataclass
class FunctionCallStat(Stat):
    call: "FunctionCall"


@dataclass
class DoBlock(Stat):
    block: Block


@dataclass
class While(Stat):
    test: "Exp"
    block: Block


@dataclass
class Repeat(Stat):
    block: Block
    test: "Exp"


@dataclass
class If(Stat):
    branches: List[tuple["Exp", Block]]  # [(cond, block), ...]
    else_block: Optional[Block] = None


@dataclass
class ForNum(Stat):
    var: Token
    start: "Exp"
    stop: "Exp"
    step: Optional["Exp"]
    block: Block


@dataclass
class ForIn(Stat):
    vars: List[Token]
    iter: List["Exp"]
    block: Block


@dataclass
class FunctionDef(Stat):
    name: "FuncName"
    body: "FuncBody"
    is_local: bool = False


@dataclass
class LocalVar(Stat):
    names: List[Tuple[Token, Optional[Token]]]
    values: Optional[List["Exp"]] = None


@dataclass
class LastStat(Stat):
    pass


@dataclass
class Return(LastStat):
    values: List["Exp"]


@dataclass
class Break(LastStat):
    pass


@dataclass
class Goto(Stat):
    label: Token


@dataclass
class Label(Stat):
    label: Token


@dataclass
class Exp(Node):
    pass


@dataclass
class Nil(Exp):
    pass


@dataclass
class Boolean(Exp):
    value: bool


@dataclass
class Number(Exp):
    value: float


@dataclass
class String(Exp):
    value: str


@dataclass
class VarArg(Exp):
    pass  # `...`


@dataclass
class TableConstructor(Exp):
    fields: List["Field"]


@dataclass
class UnaryOp(Exp):
    op: Token  # '-', 'not', '#'
    value: Exp


@dataclass
class BinaryOp(Exp):
    left: Exp
    op: Token
    right: Exp


@dataclass
class FunctionExp(Exp):
    body: "FuncBody"


@dataclass
class PrefixExp(Exp):
    expr: Union["Var", "FunctionCall", "Exp"]  # e.g. (exp)


@dataclass
class Var(Exp):
    name: str
    prefix: Optional["PrefixExp"] = None
    index: Optional[Exp] = None  # for x[exp]
    attr: Optional[str] = None  # for x.y


@dataclass
class IndexExp(Exp):
    value: Exp
    index: Exp


@dataclass
class AttrExp(Exp):
    value: Exp
    attr: Token


@dataclass
class FuncName(Node):
    names: List[Token]  # e.g. a.b.c
    method: Optional[Token] = None  # for :name


@dataclass
class FuncBody(Node):
    params: List[Token]
    is_vararg: bool
    block: Block


@dataclass
class FunctionCall(Exp):
    func: Exp
    args: List[Node]  # or TableConstructor/String for table call / string call


@dataclass
class Field(Node):
    key: Optional[Exp]  # None for list-style field
    value: Exp


def print_ast_node(node, indent=0, is_last=True, prefix=""):
    """
    Pretty-print an AST node with hierarchical structure.

    Args:
        node: The AST node to print
        indent: Current indentation level
        is_last: Whether this is the last child in a list
        prefix: Prefix string for the current line
    """
    if node is None:
        return

    # Create the connector symbols
    connector = "└── " if is_last else "├── "
    current_prefix = prefix + connector
    next_prefix = prefix + ("    " if is_last else "│   ")

    # Get the node type name
    node_type = node.__class__.__name__

    # Print the node with appropriate formatting based on type
    if isinstance(node, Block):
        print(f"{current_prefix}Block")
        for i, stat in enumerate(node.stats):
            print_ast_node(
                stat,
                indent + 1,
                i == len(node.stats) - 1 and node.laststat is None,
                next_prefix,
            )
        if node.laststat:
            print_ast_node(node.laststat, indent + 1, True, next_prefix)

    elif isinstance(node, Assignment):
        print(f"{current_prefix}Assignment")
        print(f"{next_prefix}├── Targets:")
        for i, target in enumerate(node.targets):
            print_ast_node(
                target, indent + 2, i == len(node.targets) - 1, next_prefix + "│   "
            )
        print(f"{next_prefix}└── Values:")
        for i, value in enumerate(node.values):
            print_ast_node(
                value, indent + 2, i == len(node.values) - 1, next_prefix + "    "
            )

    elif isinstance(node, FunctionCallStat):
        print(f"{current_prefix}FunctionCallStat")
        print_ast_node(node.call, indent + 1, True, next_prefix)

    elif isinstance(node, DoBlock):
        print(f"{current_prefix}DoBlock")
        print_ast_node(node.block, indent + 1, True, next_prefix)

    elif isinstance(node, While):
        print(f"{current_prefix}While")
        print(f"{next_prefix}├── Test:")
        print_ast_node(node.test, indent + 2, False, next_prefix + "│   ")
        print(f"{next_prefix}└── Block:")
        print_ast_node(node.block, indent + 2, True, next_prefix + "    ")

    elif isinstance(node, Repeat):
        print(f"{current_prefix}Repeat")
        print(f"{next_prefix}├── Block:")
        print_ast_node(node.block, indent + 2, False, next_prefix + "│   ")
        print(f"{next_prefix}└── Test:")
        print_ast_node(node.test, indent + 2, True, next_prefix + "    ")

    elif isinstance(node, If):
        print(f"{current_prefix}If")
        for i, (cond, block) in enumerate(node.branches):
            print(
                f"{next_prefix}{'├──' if i < len(node.branches) - 1 or node.else_block else '└──'} Branch {i + 1}:"
            )
            branch_prefix = next_prefix + (
                "│   " if i < len(node.branches) - 1 or node.else_block else "    "
            )
            print(f"{branch_prefix}├── Condition:")
            print_ast_node(cond, indent + 3, False, branch_prefix + "│   ")
            print(f"{branch_prefix}└── Block:")
            print_ast_node(block, indent + 3, True, branch_prefix + "    ")
        if node.else_block:
            print(f"{next_prefix}└── Else:")
            print_ast_node(node.else_block, indent + 2, True, next_prefix + "    ")

    elif isinstance(node, ForNum):
        print(f"{current_prefix}ForNum (var: {node.var})")
        print(f"{next_prefix}├── Start:")
        print_ast_node(node.start, indent + 2, False, next_prefix + "│   ")
        print(f"{next_prefix}├── Stop:")
        print_ast_node(node.stop, indent + 2, False, next_prefix + "│   ")
        if node.step:
            print(f"{next_prefix}├── Step:")
            print_ast_node(node.step, indent + 2, False, next_prefix + "│   ")
        print(f"{next_prefix}└── Block:")
        print_ast_node(node.block, indent + 2, True, next_prefix + "    ")

    elif isinstance(node, ForIn):
        print(
            f"{current_prefix}ForIn (vars: {', '.join([v.lexeme for v in node.vars])})"
        )
        print(f"{next_prefix}├── Iterators:")
        for i, iter_exp in enumerate(node.iter):
            print_ast_node(
                iter_exp, indent + 2, i == len(node.iter) - 1, next_prefix + "│   "
            )
        print(f"{next_prefix}└── Block:")
        print_ast_node(node.block, indent + 2, True, next_prefix + "    ")

    elif isinstance(node, FunctionDef):
        locality = "Local" if node.is_local else "Global"
        print(f"{current_prefix}{locality} FunctionDef")
        print_ast_node(node.name, indent + 1, False, next_prefix)
        print_ast_node(node.body, indent + 1, True, next_prefix)

    elif isinstance(node, LocalVar):
        print(
            f"{current_prefix}LocalVar (names: {', '.join([f'{n[0].lexeme}{"<" + n[1].lexeme + ">" if n[1] else ""}' for n in node.names])})"
        )
        if node.values:
            print(f"{next_prefix}└── Values:")
            for i, value in enumerate(node.values):
                print_ast_node(
                    value, indent + 2, i == len(node.values) - 1, next_prefix + "    "
                )

    elif isinstance(node, Return):
        print(f"{current_prefix}Return")
        for i, value in enumerate(node.values):
            print_ast_node(value, indent + 1, i == len(node.values) - 1, next_prefix)

    elif isinstance(node, Break):
        print(f"{current_prefix}Break")

    elif isinstance(node, Nil):
        print(f"{current_prefix}Nil")

    elif isinstance(node, Boolean):
        print(f"{current_prefix}Boolean: {node.value}")

    elif isinstance(node, Number):
        print(f"{current_prefix}Number: {node.value}")

    elif isinstance(node, String):
        print(f"{current_prefix}String: '{node.value}'")

    elif isinstance(node, VarArg):
        print(f"{current_prefix}VarArg (...)")

    elif isinstance(node, TableConstructor):
        print(f"{current_prefix}TableConstructor")
        for i, field in enumerate(node.fields):
            print_ast_node(field, indent + 1, i == len(node.fields) - 1, next_prefix)

    elif isinstance(node, UnaryOp):
        print(f"{current_prefix}UnaryOp: {node.op.lexeme}")
        print_ast_node(node.value, indent + 1, True, next_prefix)

    elif isinstance(node, BinaryOp):
        print(f"{current_prefix}BinaryOp: {node.op.kind}")
        print(f"{next_prefix}├── Left:")
        print_ast_node(node.left, indent + 2, False, next_prefix + "│   ")
        print(f"{next_prefix}└── Right:")
        print_ast_node(node.right, indent + 2, True, next_prefix + "    ")

    elif isinstance(node, FunctionExp):
        print(f"{current_prefix}FunctionExp")
        print_ast_node(node.body, indent + 1, True, next_prefix)

    elif isinstance(node, PrefixExp):
        print(f"{current_prefix}PrefixExp")
        print_ast_node(node.expr, indent + 1, True, next_prefix)

    elif isinstance(node, Var):
        if node.prefix is None:
            print(f"{current_prefix}Var: {node.name}")
        else:
            print(f"{current_prefix}Var: {node.name}")
            print(f"{next_prefix}├── Prefix:")
            print_ast_node(
                node.prefix,
                indent + 2,
                node.index is None and node.attr is None,
                next_prefix + "│   ",
            )
            if node.index:
                print(f"{next_prefix}├── Index:")
                print_ast_node(
                    node.index, indent + 2, node.attr is None, next_prefix + "│   "
                )
            if node.attr:
                print(f"{next_prefix}└── Attr: {node.attr}")

    elif isinstance(node, IndexExp):
        print(f"{current_prefix}IndexExp")
        print(f"{next_prefix}├── Value:")
        print_ast_node(
            node.value,
            indent + 2,
            True,
            next_prefix + "│   ",
        )
        print(f"{next_prefix}└── Index:")
        print_ast_node(
            node.index,
            indent + 2,
            True,
            next_prefix + "    ",
        )

    elif isinstance(node, AttrExp):
        print(f"{current_prefix}AttrExp")
        print(f"{next_prefix}├── Value:")
        print_ast_node(
            node.value,
            indent + 2,
            True,
            next_prefix + "│   ",
        )
        print(f"{next_prefix}└── Attr: {node.attr.lexeme}")

    elif isinstance(node, FuncName):
        if node.method:
            print(
                f"{current_prefix}FuncName: {'.'.join([n.lexeme for n in node.names])}:{node.method.lexeme}"
            )
        else:
            print(
                f"{current_prefix}FuncName: {'.'.join([n.lexeme for n in node.names])}"
            )

    elif isinstance(node, FuncBody):
        vararg_str = "..." if node.is_vararg else "no"
        print(
            f"{current_prefix}FuncBody (params: {', '.join((p.lexeme for p in node.params))}, vararg: {vararg_str})"
        )
        print_ast_node(node.block, indent + 1, True, next_prefix)

    elif isinstance(node, FunctionCall):
        print(f"{current_prefix}FunctionCall")
        print(f"{next_prefix}├── Function:")
        print_ast_node(node.func, indent + 2, False, next_prefix + "│   ")
        print(f"{next_prefix}└── Args:")
        for i, arg in enumerate(node.args):
            print_ast_node(
                arg, indent + 2, i == len(node.args) - 1, next_prefix + "    "
            )

    elif isinstance(node, Field):
        if node.key is None:
            print(f"{current_prefix}Field [list]:")
        else:
            print(f"{current_prefix}Field:")
            print(f"{next_prefix}├── Key:")
            print_ast_node(node.key, indent + 2, False, next_prefix + "│   ")
        print(f"{next_prefix}└── Value:")
        print_ast_node(node.value, indent + 2, True, next_prefix + "    ")

    elif isinstance(node, Goto):
        print(f"{current_prefix}Goto [{node.label.lexeme}]")

    elif isinstance(node, Label):
        print(f"{current_prefix}Label [{node.label.lexeme}]")

    else:
        print(f"{current_prefix}Unknown node: {node_type}")


# Convenience function for easier usage
def print_ast(node):
    """Print the AST starting from the given node."""
    print_ast_node(node)
