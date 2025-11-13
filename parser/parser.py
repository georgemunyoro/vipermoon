from parser.ast import (
    Assignment,
    AttrExp,
    BinaryOp,
    Block,
    Boolean,
    Break,
    DoBlock,
    Exp,
    Field,
    ForIn,
    ForNum,
    FuncBody,
    FuncName,
    FunctionCall,
    FunctionDef,
    FunctionExp,
    Goto,
    If,
    IndexExp,
    Label,
    LocalVar,
    Nil,
    Node,
    Number,
    Repeat,
    Return,
    Stat,
    String,
    TableConstructor,
    UnaryOp,
    Var,
    VarArg,
    While,
)
from parser.error import CompileError
from parser.location import Location, Span
from parser.scanner.scanner import Scanner
from parser.scanner.token import Token, TokenKind
from typing import List, Optional


class Parser:
    current: Optional[Token]

    def __init__(self, source: str):
        self.source = source
        self.tokens = iter(Scanner(source))
        self.current = None
        self.advance()

    def advance(self):
        try:
            self.current = next(self.tokens)
        except StopIteration:
            self.current = None

    def peek(self) -> Token | None:
        return self.current

    def fpeek(self) -> Token:
        if token := self.peek():
            return token
        raise Exception("unexpected end of token stream")

    def consume(self) -> Token | None:
        token = self.current
        self.advance()
        return token

    def fconsume(self) -> Token:
        if token := self.consume():
            return token
        raise Exception("unexpected end of token stream")

    def match(self, kind, value=None) -> bool:
        if self.current is None:
            return False
        if self.current.kind != kind:
            return False
        if value is not None and self.current.lexeme != value:
            return False
        self.advance()
        return True

    def check(self, kind, value=None) -> bool:
        if self.current is None:
            return False
        if self.current.kind != kind:
            return False
        if value is not None and self.current.lexeme != value:
            return False
        return True

    def expect(self, kind, value=None) -> Token:
        if not self.check(kind, value):
            err = CompileError(
                message=f"expected {kind} {value}, got {'"' + self.current.lexeme + '"' if self.current else 'EOF'}",
            )
            if self.current:
                err.span = Span(self.current.start, self.current.end)
                err.source_line = self.tokens.lines[self.current.start.line - 1]
            raise err
        return self.fconsume()

    def parse_chunk(self, at: Location = Location(0, 1, 1)):
        if not self.current:
            return Block(Span(at, at), [])

        statements = []
        while self.current and not (
            self.check(TokenKind.END) or self.check(TokenKind.UNTIL)
        ):
            statements.append(self.parse_stat())
            if self.check(TokenKind.SEMICOLON):
                self.consume()

            if self.check(TokenKind.BREAK):
                token = self.fconsume()
                statements.append(Break(Span(token.start, token.end)))
                if self.check(TokenKind.SEMICOLON):
                    self.consume()
                break

            if self.check(TokenKind.RETURN):
                ret = self.fconsume()
                explist = []
                while True:
                    explist.append(self.parse_exp())
                    if not self.match(TokenKind.COMMA):
                        break
                end = explist[-1].span.end if len(explist) > 0 else ret.end
                statements.append(
                    Return(
                        Span(ret.start, end),
                        explist,
                    )
                )
                if self.check(TokenKind.SEMICOLON):
                    self.consume()
                break

        span_end = statements[-1].span.end if len(statements) > 0 else at
        return Block(Span(at, span_end), statements)

    def parse_exp(self) -> Exp:
        if (
            self.check(TokenKind.MINUS)
            or self.check(TokenKind.NOT)
            or self.check(TokenKind.LENGTH)
        ):
            op = self.fconsume()
            exp = self.parse_exp()
            return UnaryOp(Span(op.start, exp.span.end), op, exp)

        token = self.fpeek()
        node = None

        span = Span(token.start, token.end)

        match (token.kind, token.lexeme):
            case (TokenKind.NIL, _):
                node = Nil(span)

            case (TokenKind.FALSE, _):
                node = Boolean(span, False)

            case (TokenKind.TRUE, _):
                node = Boolean(span, True)

            case (TokenKind.NUMBER, _):
                node = Number(span, float(token.lexeme))

            case (TokenKind.STRING, _):
                node = String(span, token.lexeme)

            case (TokenKind.VARARG, _):
                node = VarArg(span)

            case (TokenKind.FUNCTION, _):
                self.consume()
                body = self.parse_funcbody()
                node = FunctionExp(Span(token.start, body.span.end), body)
                node.span = Span(token.start, body.span.end)
                return node

            case (TokenKind.LBRACE, _):
                self.consume()

                fieldlist = [] if self.check(TokenKind.RBRACE) else [self.parse_field()]

                while self.check(TokenKind.COMMA) or self.check(TokenKind.SEMICOLON):
                    self.advance()
                    fieldlist.append(self.parse_field())

                if self.check(TokenKind.COMMA) or self.check(TokenKind.SEMICOLON):
                    self.advance()

                rbrace = self.expect(TokenKind.RBRACE)

                return TableConstructor(Span(token.start, rbrace.end), fieldlist)

        if node:
            self.advance()
            node.span = Span(token.start, token.end)
        else:
            node = self.parse_prefixexp()

        while self.current:
            match self.current.kind:
                case (
                    TokenKind.PLUS
                    | TokenKind.MINUS
                    | TokenKind.LT
                    | TokenKind.GT
                    | TokenKind.EQ
                ):
                    op = self.fconsume()
                    r = self.parse_exp()
                    node = BinaryOp(Span(node.span.start, r.span.end), node, op, r)

                case _:
                    break

        return node

    def parse_field(self):
        token = self.fpeek()
        if self.match(TokenKind.LBRACKET):
            key = self.parse_exp()
            self.expect(TokenKind.RBRACKET)
            self.expect(TokenKind.ASSIGN)
            value = self.parse_exp()
            return Field(Span(token.start, value.span.end), key, value)

        elif self.match(TokenKind.IDENTIFIER):
            key = Var(Span(token.start, token.end), token.lexeme)
            self.expect(TokenKind.ASSIGN)
            value = self.parse_exp()
            return Field(Span(token.start, value.span.end), key, value)

        else:
            value = self.parse_exp()
            return Field(value.span, None, value)

    def parse_funcbody(self) -> FuncBody:
        token = self.fpeek()
        self.expect(TokenKind.LPAREN)
        parlist: List[Token] = []

        if not self.check(TokenKind.RPAREN):
            if self.check(TokenKind.VARARG):
                parlist.append(self.fconsume())
            else:
                parlist = [self.expect(TokenKind.IDENTIFIER)]
                while self.match(TokenKind.COMMA):
                    parlist.append(self.fconsume())

        block = self.parse_chunk(self.expect(TokenKind.RPAREN).start)
        block.span.end = self.expect(TokenKind.END).end

        without_vararg = [p for p in parlist if p.kind != TokenKind.VARARG]
        return FuncBody(
            Span(token.start, block.span.end),
            without_vararg,
            len(without_vararg) < len(parlist),
            block,
        )

    def parse_stat(self) -> Stat:
        token = self.fpeek()
        span = Span(token.start, token.end)

        if self.match(TokenKind.COLON) and self.check(TokenKind.COLON):
            self.advance()
            label = self.expect(TokenKind.IDENTIFIER)
            self.expect(TokenKind.COLON)
            end = self.expect(TokenKind.COLON)
            return Label(Span(token.start, end.end), label)

        if self.match(TokenKind.BREAK):
            return Break(span)

        if self.match(TokenKind.GOTO):
            label = self.expect(TokenKind.IDENTIFIER)
            return Goto(Span(token.start, label.end), label)

        if self.match(TokenKind.RETURN):
            explist = []
            while True:
                explist.append(self.parse_exp())
                if not self.match(TokenKind.COMMA):
                    break
            end = explist[-1].span.end if len(explist) > 0 else token.end
            if self.check(TokenKind.SEMICOLON):
                end = self.fconsume().end
            return Return(Span(token.start, end), explist)

        if self.match(TokenKind.DO):
            block = self.parse_chunk(token.start)
            block.span.end = self.expect(TokenKind.END).end
            return DoBlock(block.span, block)

        if self.match(TokenKind.WHILE):
            condition = self.parse_exp()
            block = self.parse_chunk(self.expect(TokenKind.DO).start)
            block.span.end = self.expect(TokenKind.END).end
            return While(Span(token.start, block.span.end), condition, block)

        if self.match(TokenKind.REPEAT):
            block = self.parse_chunk(token.start)
            block.span.end = self.expect(TokenKind.UNTIL).end
            condition = self.parse_exp()
            return Repeat(Span(token.start, condition.span.end), block, condition)

        if self.match(TokenKind.IF):
            condition = self.parse_exp()
            then_token = self.expect(TokenKind.THEN)

            branches = []
            else_block = None

            statements = []
            while self.current and not (
                self.check(TokenKind.ELSEIF)
                or self.check(TokenKind.ELSE)
                or self.check(TokenKind.END)
            ):
                statements.append(self.parse_stat())

            branches.append(
                (
                    condition,
                    Block(Span(then_token.start, self.fpeek().start), statements),
                )
            )

            while self.check(TokenKind.ELSEIF):

                elseif_token = self.expect(TokenKind.ELSEIF)
                elseif_cond = self.parse_exp()
                self.expect(TokenKind.THEN)
                elseif_statements = []
                while self.current and not (
                    self.check(TokenKind.ELSEIF)
                    or self.check(TokenKind.ELSE)
                    or self.check(TokenKind.END)
                ):
                    elseif_statements.append(self.parse_stat())

                branches.append(
                    (
                        elseif_cond,
                        Block(
                            Span(elseif_token.start, self.fpeek().start),
                            elseif_statements,
                        ),
                    )
                )

            if self.check(TokenKind.ELSE):
                else_block = self.parse_chunk(self.expect(TokenKind.ELSE).start)
                else_block.span.end = self.fpeek().start

            end = self.expect(TokenKind.END)

            return If(Span(token.start, end.end), branches, else_block)

        if self.match(TokenKind.FUNCTION):
            name = self.parse_funcname()
            body = self.parse_funcbody()
            return FunctionDef(
                Span(token.start, body.span.end),
                name,
                body,
                False,
            )

        if self.match(TokenKind.LOCAL):
            if self.match(TokenKind.FUNCTION):
                ident = self.expect(TokenKind.IDENTIFIER)
                name = FuncName(Span(ident.start, ident.end), [ident])
                body = self.parse_funcbody()
                return FunctionDef(
                    Span(token.start, body.span.end),
                    name,
                    body,
                    True,
                )

            names = []

            while True:
                name = self.expect(TokenKind.IDENTIFIER)
                if self.match(TokenKind.LT):
                    names.append((name, self.expect(TokenKind.IDENTIFIER)))
                    self.expect(TokenKind.GT)
                else:
                    names.append((name, None))

                if not self.match(TokenKind.COMMA):
                    break

            explist = None
            if self.match(TokenKind.ASSIGN):
                explist = []
                while True:
                    explist.append(self.parse_exp())
                    if not self.match(TokenKind.COMMA):
                        break

            span = Span(
                token.start,
                (
                    explist[-1].span.end
                    if explist
                    else [i for i in names[-1] if i][-1].end
                ),
            )

            return LocalVar(span, names, explist)

        if self.match(TokenKind.FOR):
            vars = []
            while True:
                vars.append(self.expect(TokenKind.IDENTIFIER))
                if not self.match(TokenKind.COMMA):
                    break

            if len(vars) > 1 or self.match(TokenKind.ASSIGN):
                var: Token = vars[0]
                start = self.parse_exp()
                self.expect(TokenKind.COMMA)
                stop = self.parse_exp()
                step = None
                if self.match(TokenKind.COMMA):
                    step = self.parse_exp()
                do = self.expect(TokenKind.DO)
                block = self.parse_chunk()
                end = self.expect(TokenKind.END)

                return ForNum(
                    Span(token.start, end.end),
                    var,
                    start,
                    stop,
                    step,
                    block,
                )

            self.expect(TokenKind.IN)
            iter = []
            while True:
                iter.append(self.parse_exp())
                if not self.match(TokenKind.COMMA):
                    break
            block = self.parse_chunk(self.expect(TokenKind.DO).start)
            block.span.end = self.expect(TokenKind.END).end

            return ForIn(
                Span(token.start, block.span.end),
                vars,
                iter,
                block,
            )

        varlist = []
        while True:
            varlist.append(self.parse_prefixexp())
            if not self.match(TokenKind.COMMA):
                break

        if self.match(TokenKind.ASSIGN):
            explist = []
            while True:
                explist.append(self.parse_exp())
                if not self.match(TokenKind.COMMA):
                    break
            span = Span(token.start, explist[-1].span.end)
            return Assignment(span, varlist, explist)

        elif len(varlist) == 1:
            return varlist[0]
        else:
            raise CompileError("expected statement")

    def parse_prefixexp(self):
        token = self.fpeek()
        node = None

        if self.match(TokenKind.IDENTIFIER):
            node = Var(Span(token.start, token.end), token.lexeme)

        elif self.match(TokenKind.LPAREN):
            node = self.parse_exp()
            self.expect(TokenKind.RPAREN)

        else:
            raise CompileError(f"expected variable or '(' exp ')', got {self.current}")

        while True:
            if self.match(TokenKind.LBRACKET):
                index = self.parse_exp()
                rbrack = self.expect(TokenKind.RBRACKET)
                node = IndexExp(Span(token.start, rbrack.end), node, index)

            elif self.match(TokenKind.DOT):
                index = self.expect(TokenKind.IDENTIFIER)
                node = AttrExp(Span(node.span.start, index.end), node, index)

            elif self.match(TokenKind.COLON):
                ident = self.expect(TokenKind.IDENTIFIER)

            elif self.match(TokenKind.LPAREN):
                explist = []
                if not self.check(TokenKind.RPAREN):
                    while True:
                        explist.append(self.parse_exp())
                        if not self.match(TokenKind.COMMA):
                            break
                rparen = self.expect(TokenKind.RPAREN)
                node = FunctionCall(Span(node.span.start, rparen.end), node, explist)

            elif self.check(TokenKind.STRING):
                token = self.expect(TokenKind.STRING)
                explist: List[Node] = [
                    String(Span(token.start, token.end), token.lexeme)
                ]
                node = FunctionCall(Span(node.span.start, token.end), node, explist)

            elif self.check(TokenKind.LBRACE):
                raise Exception("todo")

            else:
                break

        return node

    def parse_funcname(self):
        names = [self.expect(TokenKind.IDENTIFIER)]
        method = None
        while self.match(TokenKind.DOT):
            names.append(self.expect(TokenKind.IDENTIFIER))
        if self.match(TokenKind.COLON):
            method = self.expect(TokenKind.IDENTIFIER)

        span = Span(
            start=names[0].start,
            end=method.end if method else names[len(names) - 1].end,
        )

        return FuncName(span, names, method)
