import parser.ast as ast

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

    def set_last_instr_sourceline(self, line: int):
        self.proto.instructions[len(self.proto.instructions) - 1].sourceline = line


class CodegenContext:
    def __init__(self, proto: Prototype):
        self.proto = proto
        self.reg_top = 0
        self.max_stack_used = 0
        self.constants = ConstantTable(proto)
        self.emitter = InstructionEmitter(proto)
        self.blocks = []  # for loop breaks, etc.
        self.expected_return_values = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.emitter.emit_ABC(Op.RETURN, 0, 1, 0)
        self.proto.max_stack_size = self.max_stack_used

    # Register management
    def alloc_reg(self):
        r = self.reg_top
        self.reg_top += 1
        self.max_stack_used = max(self.max_stack_used, self.reg_top)
        return r

    def free_reg(self, r):
        if r == self.reg_top - 1:
            self.reg_top -= 1


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

    with CodegenContext(proto) as ctx:
        for stat in node.stats:
            compile_node(ctx, stat)

    return proto


def compile_node(ctx: CodegenContext, node: ast.Node):
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
        )  # BUG: handle return value count
        ctx.emitter.set_last_instr_sourceline(node.span.start.line)

        return rA

    elif isinstance(node, ast.Var):
        r = ctx.alloc_reg()
        k = ctx.constants.get(Constant(Constant.Kind.STRING, node.name))
        ctx.emitter.emit_ABx(Op.GETGLOBAL, r, k)
        ctx.emitter.set_last_instr_sourceline(node.span.start.line)
        return r

    elif isinstance(node, ast.FuncName):
        assert len(node.names) == 1
        r = ctx.alloc_reg()
        k = ctx.constants.get(Constant(Constant.Kind.STRING, node.names[0].lexeme))
        ctx.emitter.emit_ABx(Op.GETGLOBAL, r, k)
        ctx.emitter.set_last_instr_sourceline(node.span.start.line)
        return r

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

        r = ctx.alloc_reg()
        ctx.emitter.emit_ABx(Op.CLOSURE, r, pidx)
        ctx.emitter.set_last_instr_sourceline(node.span.start.line)

        assert len(node.name.names) == 1
        fn_name = node.name.names[0]

        if not node.is_local:
            ctx.emitter.emit_ABx(
                Op.SETGLOBAL,
                r,
                ctx.constants.get(
                    Constant(
                        Constant.Kind.STRING,
                        fn_name.lexeme,
                    )
                ),
            )
            ctx.emitter.set_last_instr_sourceline(node.span.start.line)
            ctx.free_reg(r)

        return r

    elif isinstance(node, ast.Return):
        if len(node.values) == 0:
            ctx.emitter.emit_ABC(Op.RETURN, 0, 1)
            ctx.emitter.set_last_instr_sourceline(node.span.start.line)
            return

        regs = [compile_node(ctx, value) for value in node.values]
        first_reg = regs[0]
        num_values = len(regs) + 1  # B = number of values + 1
        ctx.emitter.emit_ABC(Op.RETURN, first_reg, num_values)
        ctx.emitter.set_last_instr_sourceline(node.span.start.line)

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
        for target in node.targets:
            value_reg = value_regs[i] + j

            if isinstance(target, ast.Var):
                ctx.emitter.emit_ABx(
                    Op.SETGLOBAL,
                    value_reg,
                    ctx.constants.get(Constant(Constant.Kind.STRING, target.name)),
                )
                ctx.emitter.set_last_instr_sourceline(node.span.start.line)
                # ctx.free_reg(value_reg)

            else:
                raise Exception(f"invalid/unhandled assignment target: {target}")

            if (i + 1) < len(value_regs):
                i += 1
            else:
                j += 1

    elif isinstance(node, ast.String):
        r = ctx.alloc_reg()
        k = ctx.constants.get(Constant(Constant.Kind.STRING, node.value))
        ctx.emitter.emit_ABx(Op.LOADK, r, k)
        ctx.emitter.set_last_instr_sourceline(node.span.start.line)
        return r

    elif isinstance(node, ast.Number):
        r = ctx.alloc_reg()
        k = ctx.constants.get(
            Constant(
                Constant.Kind.NUMBER,
                int(node.value) if node.value.is_integer() else node.value,
            )
        )
        ctx.emitter.emit_ABx(Op.LOADK, r, k)
        ctx.emitter.set_last_instr_sourceline(node.span.start.line)
        return r

    else:
        raise Exception(f"encountered unknown node: {node}")
