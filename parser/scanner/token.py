import enum
from dataclasses import dataclass

from ..location import Location


class TokenKind(enum.Enum):
    # --- Special ---
    EOF = "EOF"
    UNKNOWN = "UNKNOWN"

    # --- Identifiers and literals ---
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"

    # --- Keywords ---
    AND = "and"
    BREAK = "break"
    DO = "do"
    ELSE = "else"
    ELSEIF = "elseif"
    END = "end"
    FALSE = "false"
    FOR = "for"
    FUNCTION = "function"
    GOTO = "goto"
    IF = "if"
    IN = "in"
    LOCAL = "local"
    NIL = "nil"
    NOT = "not"
    OR = "or"
    REPEAT = "repeat"
    RETURN = "return"
    THEN = "then"
    TRUE = "true"
    UNTIL = "until"
    WHILE = "while"

    # --- Operators ---
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    FLOOR_DIV = "//"
    MOD = "%"
    POW = "^"
    CONCAT = ".."
    LENGTH = "#"

    EQ = "=="
    NE = "~="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="

    ASSIGN = "="

    # --- Delimiters ---
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    SEMICOLON = ";"
    COLON = ":"
    COMMA = ","
    DOT = "."
    VARARG = "..."  # for function varargs

    # --- Comments (optional to tokenize explicitly) ---
    COMMENT = "COMMENT"


@dataclass
class Token:
    kind: TokenKind
    start: Location
    end: Location
    lexeme: str
