import parser.ast as ast
import struct
from typing import List

from .constant import Constant
from .instruction import Instruction, Op, iABx
from .prototype import Prototype


class ConstantTable:
    def __init__(self, proto):
        self.proto = proto
        self.lookup = {}  # value → index

    def get(self, value: Constant):
        if value.const in self.lookup:
            return self.lookup[value]
        idx = len(self.proto.constants)
        self.lookup[value.const] = idx
        self.proto.constants.append(value)
        return idx


class InstructionEmitter:
    def __init__(self, proto):
        self.proto = proto

    def emit_ABC(self, op, a, b, c):
        ins = (op.value & 0x3F) | (a << 6) | (b << 23) | (c << 14)
        self.proto.instructions.append(Instruction(ins))

    def emit_ABx(self, op, a, bx):
        ins = (op.value & 0x3F) | (a << 6) | (bx << 14)
        self.proto.instructions.append(Instruction(ins))

    def emit_RETURN(self, a):
        # B=1: return 1 value
        self.emit_ABC(Op.RETURN, a, 2, 0)


class CodegenContext:
    def __init__(self, proto: Prototype):
        self.proto = proto
        self.reg_top = 0
        self.max_stack_used = 0
        self.constants = ConstantTable(proto)
        self.emitter = InstructionEmitter(proto)
        self.blocks = []  # for loop breaks, etc.

    # Register management
    def alloc_reg(self):
        r = self.reg_top
        self.reg_top += 1
        self.max_stack_used = max(self.max_stack_used, self.reg_top)
        return r

    def free_reg(self, r):
        if r == self.reg_top - 1:
            self.reg_top -= 1

    # Emitting
    def emit(self, op, a=0, b=0, c=0):
        self.emitter.emit_ABC(op, a, b, c)

    def emit_return(self, reg=0):
        self.emitter.emit_RETURN(reg)


def compile(node: ast.Block, source="@test.lua") -> Prototype:
    proto = Prototype(
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

    ctx = CodegenContext(proto)

    for stat in node.stats:
        compile_stat(ctx, stat)

    ctx.emitter.emit_ABC(Op.RETURN, 0, 1, 0)

    proto.max_stack_size = ctx.max_stack_used
    return proto


def compile_stat(ctx: CodegenContext, stat: ast.Node):
    if isinstance(stat, ast.FunctionCall):
        if stat.method:
            raise Exception("method calls not implemented")

        fn = ctx.alloc_reg()

        if isinstance(stat.func, ast.Var):
            k = ctx.constants.get(Constant(Constant.Kind.STRING, stat.func.name))
            ctx.emitter.emit_ABx(Op.GETGLOBAL, fn, k)

        arg_regs = []
        for arg in stat.args:
            arg_regs.append(compile_exp(ctx, arg))

        ctx.emitter.emit_ABC(Op.CALL, fn, len(arg_regs) + 1, 1)


def compile_exp(ctx: CodegenContext, exp: ast.Node):
    if isinstance(exp, ast.String):
        r = ctx.alloc_reg()
        k = ctx.constants.get(Constant(Constant.Kind.STRING, exp.value))
        ctx.emitter.emit_ABx(Op.LOADK, r, k)
        return r
