import re
from typing import Union

from ..error import CompileError
from ..location import Location, Span
from .token import Token, TokenKind


class Scanner:
    LEXEME_TOKEN_MAP = {k.value: k for k in TokenKind}

    def __init__(self, source: str):
        self.lines = source.splitlines()
        self.source = source
        self.length = len(source)
        self.start = 0
        self.line = 1
        self.col = 1
        self.location = Location(0, 1, 1)

        if self.source.startswith("#!"):
            while self.current and self.current != "\n":
                self.advance()

    def __iter__(self):
        return self

    def __next__(self):
        token = self.next_token
        if token is None:
            raise StopIteration()
        return token

    def expect(self, expected: str, start_location=None):
        if self.current != expected:
            raise CompileError(
                message=f"unterminated string",
                span=Span(self.location, self.location),
                source_line=self.lines[
                    (
                        start_location.line
                        if start_location is not None
                        else self.location.line
                    )
                    - 1
                ],
            )
        return self.advance()

    def is_at(self, s: str):
        if len(self.source) < (self.location.index + len(s)):
            return False
        i = self.location.index
        return self.source[i : i + len(s)] == s

    @property
    def next_token(self):
        self.skip_whitespace()

        self.start = self.location.index

        c = self.current
        if c is None:
            return None

        start = Location(self.location.index, self.location.line, self.location.col)

        match c:
            case c if c.isalpha() or c == "_":
                while self.current is not None and (
                    self.current.isalnum() or self.current == "_"
                ):
                    self.advance()

                lexeme = self.lexeme(start.index)
                kind = (
                    TokenKind.IDENTIFIER
                    if lexeme not in self.LEXEME_TOKEN_MAP
                    else self.LEXEME_TOKEN_MAP[lexeme]
                )

                return Token(
                    kind,
                    start,
                    self.location,
                    self.lexeme(start.index),
                )

            case c if c.isdigit():
                return self.scan_numeric()

            case "[":
                kind = TokenKind.LBRACKET
                self.advance()

                eq_count = 0
                if self.current and self.current == "=":
                    while self.current and self.current != "[":
                        eq_count += 1
                        self.advance()

                if self.current and self.current == "[":
                    self.advance()
                    opening_quote = self.lexeme(start.index)
                    closing_quote = opening_quote.replace("[", "]")

                    while self.current and not self.is_at(closing_quote):
                        self.advance()

                    if not self.is_at(closing_quote):
                        raise CompileError(
                            message=f"unterminated string, expected ending quote",
                            span=Span(start, self.location),
                            source_line=self.lines[start.line - 1],
                        )

                    for _ in range(len(opening_quote)):
                        self.advance()

                    kind = TokenKind.STRING

                return Token(
                    kind,
                    start,
                    self.location,
                    self.lexeme(start.index),
                )

            case "-":
                self.advance()
                if self.current is not None and self.current == "-":
                    self.advance()

                    if m := re.match(r"^\[[=]+\[", self.source[self.location.index :]):
                        opening_quote = m.group()
                        closing_quote = opening_quote.replace("[", "]")

                        while self.current and not self.is_at(closing_quote):
                            self.advance()

                        if not self.is_at(closing_quote):
                            raise CompileError(
                                message=f"unterminated multiline comment, expected ending quote",
                                span=Span(start, self.location),
                                source_line=self.lines[start.line - 1],
                            )

                        for _ in range(len(opening_quote)):
                            self.advance()

                        return self.next_token

                    while self.current is not None and self.current != "\n":
                        self.advance()

                    return self.next_token

                return Token(
                    TokenKind.MINUS,
                    start,
                    self.location,
                    self.lexeme(start.index),
                )

            case ".":

                self.advance()
                if self.current is not None and self.current == ".":
                    self.advance()
                    if self.current is not None and self.current == ".":
                        self.advance()

                lexeme = self.lexeme(start.index)
                return Token(
                    self.LEXEME_TOKEN_MAP[lexeme],
                    start,
                    self.location,
                    lexeme,
                )

            case '"' | "'":
                opening_quote = self.advance()

                lexeme = ""

                while self.current is not None and self.current != opening_quote:
                    if self.current == "\\":
                        self.advance()
                        c = self.current

                        match c:
                            case "n":
                                lexeme += "\n"
                            case "t":
                                lexeme += "\t"
                            case "\\":
                                lexeme += "\\"
                            case "a":
                                lexeme += "\a"
                            case '"':
                                lexeme += '"'
                            case "'":
                                lexeme += "'"
                            case "b":
                                lexeme += "\b"
                            case "f":
                                lexeme += "\f"
                            case "b":
                                lexeme += "\b"
                            case "f":
                                lexeme += "\f"
                            case "r":
                                lexeme += "\r"
                            case "v":
                                lexeme += "\v"
                            case "\n":
                                lexeme += "\n"

                            case c if c and c.isdigit():
                                d = f"{c}{self.advance()}"
                                lexeme += chr(int(d))

                            case _:
                                raise CompileError(
                                    message=f"invalid escape character: {c}",
                                    span=Span(self.location, self.location),
                                    source_line=self.lines[self.location.line - 1],
                                )
                    else:
                        lexeme += self.current

                    self.advance()

                if self.advance() != opening_quote:
                    raise CompileError(
                        message=f"unterminated string, expected ending quote",
                        span=Span(start, self.location),
                        source_line=self.lines[start.line - 1],
                    )

                return Token(
                    TokenKind.STRING,
                    start,
                    self.location,
                    lexeme,
                )

            case "=" | "<" | ">" | "~":
                self.advance()

                if self.current is not None and self.current == "=":
                    self.advance()

                lexeme = self.lexeme(start.index)

                return Token(
                    self.LEXEME_TOKEN_MAP[lexeme],
                    start,
                    self.location,
                    lexeme,
                )

            case (
                "{"
                | "}"
                | ","
                | "("
                | ")"
                | "#"
                | ":"
                | "]"
                | "+"
                | "/"
                | "%"
                | ";"
                | "*"
                | "^"
            ):
                self.advance()
                return Token(
                    self.LEXEME_TOKEN_MAP[c],
                    start,
                    self.location,
                    self.lexeme(start.index),
                )

        raise CompileError(
            message=f"encountered unknown character {c} at {self.location}",
            source_line=self.lines[self.location.line - 1],
            span=Span(start, self.location),
        )

    def skip_whitespace(self):
        while self.current is not None and self.current.isspace():
            self.advance()

    @property
    def is_at_end(self):
        return self.location.index > self.length

    @property
    def current(self):
        if self.location.index >= self.length:
            return None
        return self.source[self.location.index]

    def advance(self, offset=1):
        if self.current is None:
            return None

        current = self.current

        for _ in range(offset):
            if self.current is None:
                return None

            current = self.source[self.location.index]
            self.location.index += 1

            if current == "\n":
                self.location.line += 1
                self.location.col = 1
            else:
                self.location.col += 1

        return current

    def lexeme(self, start: int, end: Union[int, None] = None):
        return self.source[start : end if end is not None else self.location.index]

    def scan_numeric(self):
        start = Location(self.location.index, self.location.line, self.location.col)
        encountered_period = False
        while self.current is not None:
            if self.current == ".":
                if encountered_period:
                    break

                encountered_period = True
                self.advance()
                continue

            if not self.current.isdigit():
                break

            self.advance()

        return Token(
            TokenKind.NUMBER,
            start,
            self.location,
            self.lexeme(start.index),
        )
