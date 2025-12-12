from parser import ast
from parser.location import Location, Span
from parser.scanner.token import Token, TokenKind

from compiler.compile import fold

_l = Location(0, 0, 0)
_s = Span(_l, _l)


def test_compile_fold():
    b = fold(
        ast.BinaryOp(
            _s,
            ast.Number(_s, 10),
            Token(TokenKind.PLUS, _l, _l, "+"),
            ast.Number(_s, 20),
        )
    )
    assert isinstance(b, ast.Number)
    assert b.value == 30

    b = fold(
        ast.BinaryOp(
            _s,
            ast.Number(_s, 10),
            Token(TokenKind.MINUS, _l, _l, "+"),
            ast.Number(_s, 20),
        )
    )
    assert isinstance(b, ast.Number)
    assert b.value == -10

    b = fold(
        ast.BinaryOp(
            _s,
            ast.BinaryOp(
                _s,
                ast.Number(_s, 3),
                Token(TokenKind.MINUS, _l, _l, "-"),
                ast.Number(_s, 2),
            ),
            Token(TokenKind.PLUS, _l, _l, "+"),
            ast.BinaryOp(
                _s,
                ast.Number(_s, 4),
                Token(TokenKind.MINUS, _l, _l, "-"),
                ast.Number(_s, 2),
            ),
        )
    )
    assert isinstance(b, ast.Number)
    assert b.value == 3
