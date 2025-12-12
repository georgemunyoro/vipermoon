import parser.ast as ast
from parser.scanner.token import TokenKind
from typing import Dict, Optional

from .constant import Constant
from .instruction import Instruction, Op
from .prototype import Local, Prototype


class ConstantTable:
    def __init__(self, proto):
        self.proto = proto
        self.lookup = {}  # value → index

    def get(self, value: Constant):
        if value.const in self.lookup:
            return self.lookup[value.const]
        idx = len(self.proto.constants)
        self.lookup[value.const] = idx
        self.proto.constants.append(value)
        return idx


class InstructionEmitter:
    def __init__(self, proto):
        self.proto = proto

    def emit_ABC(self, op, a, b, c=0):
        ins = (op.value & 0x3F) | (a << 6) | (b << 23) | (c << 14)
        self.proto.instructions.append(Instruction(ins))

    def emit_ABx(self, op, a, bx):
        ins = (op.value & 0x3F) | (a << 6) | (bx << 14)
        self.proto.instructions.append(Instruction(ins))

    def emit_RETURN(self, a):
        # B=1: return 1 value
        self.emit_ABC(Op.RETURN, a, 2, 0)


class Codegen:
    def __init__(self):
        self.instructions = []
        self.constants = []
        self.reg_count = 0
        self.labels = {}

    def alloc_reg(self):
        reg = self.reg_count
        self.reg_count += 1
        return reg

    def free_reg(self):
        self.reg_count -= 1

    def get_const(self, value):
        if value in self.constants:
            return self.constants.index(value)
        self.constants.append(value)
        return len(self.constants) - 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): ...


class CodegenContext:
    def __init__(self, proto: Prototype):
        self.proto = proto
        self.reg_top = 0
        self.max_stack_used = 0
        self.constants = ConstantTable(proto)
        self.emitter = InstructionEmitter(proto)
        self.blocks = []  # for loop breaks, etc.
        self.expected_return_values = 0

        self.locals = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.emitter.emit_ABC(Op.RETURN, 0, 1, 0)
        self.proto.max_stack_size = self.max_stack_used
        self.proto.locals = list(self.locals.keys())

    def get_local(self, name: str):
        if name in self.locals:
            return self.locals[name]
        self.locals[name] = self.alloc_reg()
        return self.locals[name]

    # Register management
    def alloc_reg(self, msg: Optional[str] = None):
        r = self.reg_top

        if msg:
            print(f"ALLC({r}): {msg}")

        self.reg_top += 1
        self.max_stack_used = max(self.max_stack_used, self.reg_top)
        return r

    def free_reg(self, r, msg: Optional[str] = None):
        if msg:
            print(f"FREE({r}): {msg}")

        if r == self.reg_top - 1:
            self.reg_top -= 1


def get_main_proto(source="@test.lua"):
    return Prototype(
        source_name=source,
        line_defined=0,
        last_line_defined=0,
        num_upvalues=0,
        num_parameters=0,
        is_vararg=False,
        max_stack_size=2,
        instructions=[],
        constants=[],
        locals=[],
        source_line_position_list=[],
        upvalues=[],
        prototypes=[],
    )


def compile(node: ast.Block, source="@test.lua") -> Prototype:
    proto = get_main_proto(source)

    with CodegenContext(proto) as ctx:
        for stat in node.stats:
            compile_node(ctx, stat)

    return proto


def fold(node: ast.BinaryOp):
    l = fold(node.left) if isinstance(node.left, ast.BinaryOp) else node.left
    r = fold(node.right) if isinstance(node.right, ast.BinaryOp) else node.right

    if isinstance(l, ast.Number) and isinstance(r, ast.Number):
        match node.op.kind:
            case TokenKind.PLUS:
                return ast.Number(node.span, l.value + r.value)
            case TokenKind.MINUS:
                return ast.Number(node.span, l.value - r.value)

    return node


def compile_node(
    ctx: CodegenContext, node: ast.Node, load_comparison_value: bool = True
):
    def compile_rk_operand(n: ast.Node):
        if is_literal(n):
            return 256 + compile_literal(ctx, n)
        else:
            return compile_node(ctx, n)

    if isinstance(node, ast.FunctionCall):
        if node.method:
            raise Exception("method calls not implemented")

        rA = compile_node(ctx, node.func)

        arg_regs = []
        for arg in node.args:
            arg_regs.append(compile_node(ctx, arg))

        rB = len(node.args) + 1
        if isinstance(node.args[:-1], ast.FunctionCall):
            rB = 0

        num_return_values = ctx.expected_return_values

        ctx.emitter.emit_ABC(
            Op.CALL,
            rA,
            rB,
            num_return_values + 1,
        )  # BUG: handle return value when arg is function call

        return rA

    elif isinstance(node, ast.Var):
        try:
            locals = [l.name for l in ctx.proto.locals]
            return locals.index(node.name)
        except Exception as e:
            rvalue = ctx.alloc_reg(f"getglobal {node.name}")
            k = ctx.constants.get(Constant(Constant.Kind.STRING, node.name))
            ctx.emitter.emit_ABx(Op.GETGLOBAL, rvalue, k)
            return rvalue

    elif isinstance(node, ast.FuncName):
        assert len(node.names) == 1
        rvalue = ctx.alloc_reg(f"getglobal {node.names[0].lexeme}")
        k = ctx.constants.get(Constant(Constant.Kind.STRING, node.names[0].lexeme))
        ctx.emitter.emit_ABx(Op.GETGLOBAL, rvalue, k)
        return rvalue

    elif isinstance(node, ast.FunctionDef):
        proto = Prototype(
            source_name="",
            line_defined=node.span.start.line,
            last_line_defined=node.span.end.line,
            num_upvalues=0,
            num_parameters=len(node.body.params),
            is_vararg=node.body.is_vararg,
            max_stack_size=2,
            instructions=[],
            constants=[],
            locals=[],
            source_line_position_list=[],
            upvalues=[],
            prototypes=[],
        )

        with CodegenContext(proto) as fn_ctx:
            for param in node.body.params:
                fn_ctx.proto.locals.append(
                    Local(param.lexeme, 0, 0)
                )  # TODO: start and end pc

            for body_stat in node.body.block.stats:
                compile_node(fn_ctx, body_stat)

        pidx = len(ctx.proto.prototypes)
        ctx.proto.prototypes.append(proto)

        rvalue = ctx.alloc_reg(f"functiondef storing closure")
        ctx.emitter.emit_ABx(Op.CLOSURE, rvalue, pidx)

        assert len(node.name.names) == 1
        fn_name = node.name.names[0]

        if not node.is_local:
            ctx.emitter.emit_ABx(
                Op.SETGLOBAL,
                rvalue,
                ctx.constants.get(
                    Constant(
                        Constant.Kind.STRING,
                        fn_name.lexeme,
                    )
                ),
            )
            ctx.free_reg(rvalue, "closure stored globally, freeing")

        return rvalue

    elif isinstance(node, ast.Return):
        if len(node.values) == 0:
            ctx.emitter.emit_ABC(Op.RETURN, 0, 1)
            return

        regs = [compile_node(ctx, value) for value in node.values]
        first_reg = regs[0]
        num_values = len(regs) + 1  # B = number of values + 1
        ctx.emitter.emit_ABC(Op.RETURN, first_reg, num_values)

    elif isinstance(node, ast.LocalVar):
        values = node.values if node.values else []
        remaining_required_values = len(values)

        value_regs = []
        for value in values[:-1]:
            remaining_required_values -= 1
            ctx.expected_return_values = 1
            value_regs.append(compile_node(ctx, value))

        ctx.expected_return_values = remaining_required_values
        value_regs.append(compile_node(ctx, values[-1]))

        ctx.expected_return_values = 0

        i = 0
        j = 0
        for name in node.names:
            value_reg = value_regs[i] + j

            assert name[1] is None

            if value_reg is not None:
                ctx.free_reg(value_reg)

            reg = ctx.get_local(name[0].lexeme)
            ctx.emitter.emit_ABC(Op.MOVE, reg, value_reg)

            if (i + 1) < len(value_regs):
                i += 1
            else:
                j += 1

    elif isinstance(node, ast.Assignment):
        remaining_required_values = len(node.targets)

        value_regs = []
        for value in node.values[:-1]:
            remaining_required_values -= 1
            ctx.expected_return_values = 1
            value_regs.append(compile_node(ctx, value))

        ctx.expected_return_values = remaining_required_values
        value_regs.append(compile_node(ctx, node.values[-1]))

        ctx.expected_return_values = 0

        i = 0
        j = 0
        for name in node.targets:
            value_reg = value_regs[i] + j

            if isinstance(name, ast.Var):
                ctx.emitter.emit_ABx(
                    Op.SETGLOBAL,
                    value_reg,
                    ctx.constants.get(Constant(Constant.Kind.STRING, name.name)),
                )
                ctx.free_reg(value_reg, "global assignment rhs, saved to _G")

            else:
                raise Exception(f"invalid/unhandled assignment target: {name}")

            if (i + 1) < len(value_regs):
                i += 1
            else:
                j += 1

    elif isinstance(node, ast.If):
        for condition, block in node.branches:
            compile_node(ctx, condition, load_comparison_value=False)
            b_start = len(ctx.proto.instructions)
            compile_node(ctx, block)
            b_end = len(ctx.proto.instructions)

            jmp_ins_bx = b_end - b_start + 131071
            jmp_ins = Instruction((Op.JMP.value & 0x3F) | (0 << 6) | (jmp_ins_bx << 14))
            ctx.proto.instructions.insert(b_start, jmp_ins)

    elif isinstance(node, ast.Block):
        for stat in node.stats:
            compile_node(ctx, stat)

    elif isinstance(node, ast.BinaryOp):
        node = fold(node)
        if not isinstance(node, ast.BinaryOp):
            return compile_node(ctx, node)

        l_start = len(ctx.proto.instructions)
        lreg = compile_rk_operand(node.left)
        l_end = len(ctx.proto.instructions)
        rreg = compile_rk_operand(node.right)

        if not is_literal(node.left):
            ctx.free_reg(lreg, "lhs of == op")

        if not is_literal(node.right):
            ctx.free_reg(rreg, "rhs of == op")

        r = ctx.alloc_reg("value for storing == op result")

        token_op_map: Dict[str, Op] = {}
        token_op_map[TokenKind.EQ.value] = Op.EQ
        token_op_map[TokenKind.LE.value] = Op.LE
        token_op_map[TokenKind.MINUS.value] = Op.SUB
        token_op_map[TokenKind.PLUS.value] = Op.ADD
        op = token_op_map[node.op.kind.value]

        match node.op.kind:
            case TokenKind.EQ | TokenKind.LE:
                ctx.emitter.emit_ABC(op, r, lreg, rreg)

                if load_comparison_value:
                    ctx.emitter.emit_ABx(Op.JMP, r, max(l_end - l_start, 1) + 131071)
                    ctx.emitter.emit_ABC(Op.LOADBOOL, r, 0, 1)
                    ctx.emitter.emit_ABC(Op.LOADBOOL, r, 1, 0)

            case TokenKind.MINUS | TokenKind.PLUS:
                ctx.emitter.emit_ABC(op, r, lreg, rreg)

            case _:
                raise Exception(f"unknown/unhandled binary op: {node.op}")

        return r

    elif is_literal(node):
        rvalue = ctx.alloc_reg(f"value for storing literal {node}")
        if isinstance(node, ast.Boolean):
            ctx.emitter.emit_ABC(
                Op.LOADBOOL,
                rvalue,
                int(node.value),
                0,
            )
        else:
            k = compile_literal(ctx, node)
            ctx.emitter.emit_ABx(Op.LOADK, rvalue, k)
        return rvalue

    else:
        raise Exception(f"encountered unknown node: {node}")


def is_literal(node: ast.Node):
    return (
        isinstance(node, ast.Number)
        or isinstance(node, ast.String)
        or isinstance(node, ast.Boolean)
    )


def compile_literal(ctx: CodegenContext, node: ast.Node):
    if isinstance(node, ast.Number):
        return ctx.constants.get(
            Constant(
                Constant.Kind.NUMBER,
                int(node.value) if node.value.is_integer() else node.value,
            )
        )

    elif isinstance(node, ast.String):
        return ctx.constants.get(
            Constant(
                Constant.Kind.STRING,
                node.value,
            )
        )

    elif isinstance(node, ast.Boolean):
        return ctx.constants.get(
            Constant(
                Constant.Kind.BOOLEAN,
                node.value,
            )
        )

    else:
        raise Exception(f"invalid/unhandled literal kind: {node}")


def gen_exp(cg: Codegen, exp: ast.Exp):
    if isinstance(exp, ast.Number):
        reg = cg.alloc_reg()
        const_idx = cg.get_const(exp.value)
        return reg
