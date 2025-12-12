import parser.ast as ast
from parser.scanner.token import TokenKind
from typing import Dict, List, Union

from .constant import Constant
from .instruction import Instruction, Op
from .prototype import Local, Prototype


class Codegen:
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.instructions = []
        self.constants: List[Constant] = []
        self.reg_count = 0
        self.labels = {}
        self.locals: Dict[str, int] = {}
        self.functions = {}
        self.max_stack_used = 0

    def alloc_reg(self):
        reg = self.reg_count
        self.reg_count += 1
        self.max_stack_used = max(self.max_stack_used, self.reg_count)
        return reg

    def free_reg(self, reg: int):
        if reg == self.reg_count - 1:
            self.reg_count -= 1

    def get_const(self, value):
        if value in self.constants:
            return self.constants.index(value)

        kind = None
        if isinstance(value, str):
            kind = Constant.Kind.STRING
        elif isinstance(value, float) or isinstance(int, value):
            kind = Constant.Kind.NUMBER
        if kind is None:
            raise

        self.constants.append(Constant(kind=kind, const=value))
        return len(self.constants) - 1

    def emit(self, op: Op, a: int, b=None, c=None, bx=None, sbx=None) -> int:
        assert a is not None
        instr_idx = len(self.instructions)
        instr = Instruction(op, a, b, c, bx, sbx)
        self.instructions.append(instr)
        return instr_idx

    def set_local_reg(self, name: str, reg: int):
        self.locals[name] = reg

    def get_local_reg(self, name: str):
        if name not in self.locals:
            raise
        return self.locals[name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): ...

    def get_proto(self):
        return Prototype(
            source_name=self.source_name,
            line_defined=0,
            last_line_defined=0,
            num_upvalues=0,
            num_parameters=0,
            is_vararg=False,
            max_stack_size=max(self.max_stack_used, 2),
            instructions=self.instructions,
            constants=self.constants,
            locals=[Local(l, 0, 0) for l in self.locals.keys()],
            source_line_position_list=[],
            upvalues=[],
            prototypes=[],
        )


def gen_exp(cg: Codegen, exp: Union[ast.Exp, ast.Node]):
    if isinstance(exp, ast.Var):
        if exp.prefix is None:
            if exp.name in cg.locals:
                return cg.get_local_reg(exp.name)
            else:
                reg = cg.alloc_reg()
                const_idx = cg.get_const(exp.name)
                cg.emit(Op.GETGLOBAL, a=reg, bx=const_idx)
                return reg

    if isinstance(exp, ast.BinaryOp):

        def _gen(n: ast.Node):
            if isinstance(n, ast.Number) or isinstance(n, ast.String):
                return False, 256 + cg.get_const(n.value)
            else:
                return True, gen_exp(cg, n)

        is_l_reg, left = _gen(exp.left)
        is_r_reg, right = _gen(exp.right)

        should_switch_operands = False
        comp_opcode = None

        if exp.op.kind == TokenKind.LT:
            comp_opcode = Op.LT
        elif exp.op.kind == TokenKind.LE:
            comp_opcode = Op.LE
        elif exp.op.kind == TokenKind.GT:
            should_switch_operands = True
            comp_opcode = Op.LT
        elif exp.op.kind == TokenKind.GE:
            should_switch_operands = True
            comp_opcode = Op.LE
        elif exp.op.kind == TokenKind.EQ:
            comp_opcode = Op.EQ

        if comp_opcode:
            if should_switch_operands:
                left, right = right, left

            dest = cg.alloc_reg()
            cg.emit(comp_opcode, a=1, b=left, c=right)
            cg.emit(Op.JMP, a=0, sbx=1)
            cg.emit(Op.LOADBOOL, a=dest, b=0, c=1)
            cg.emit(Op.LOADBOOL, a=dest, b=1, c=0)
            return dest

        comp_opcode = None
        match exp.op.kind:
            case TokenKind.PLUS:
                comp_opcode = Op.ADD
            case TokenKind.STAR:
                comp_opcode = Op.MUL
            case TokenKind.MINUS:
                comp_opcode = Op.SUB
            case TokenKind.MOD:
                comp_opcode = Op.MOD
            case TokenKind.SLASH:
                comp_opcode = Op.DIV
            case TokenKind.POW:
                comp_opcode = Op.POW

        if not comp_opcode:
            print(f"invalid binary op: {exp.op.lexeme}")
            exit(1)

        dest = cg.alloc_reg()
        print("reg", cg.reg_count)

        cg.emit(comp_opcode, a=dest, b=left, c=right)
        return dest

    reg = cg.alloc_reg()

    if isinstance(exp, ast.Number) or isinstance(exp, ast.String):
        const_idx = cg.get_const(exp.value)
        cg.emit(Op.LOADK, a=reg, bx=const_idx)

    elif isinstance(exp, ast.Boolean):
        cg.emit(Op.LOADBOOL, a=reg, b=1 if exp.value else 0, c=0)

    elif isinstance(exp, ast.Nil):
        cg.emit(Op.LOADNIL, a=reg, b=reg)

    else:
        print("unhandled exp: ", exp)
        exit(1)

    return reg


def gen_function_call(cg: Codegen, call: ast.FunctionCall, num_results=0):
    arg_regs = [gen_exp(cg, arg) for arg in call.args]
    func_reg = gen_exp(cg, call.func)

    for (
        i,
        reg,
    ) in enumerate(arg_regs):
        cg.emit(Op.MOVE, cg.alloc_reg(), reg)

    cg.emit(Op.CALL, a=func_reg, b=len(arg_regs) + 1, c=num_results + 1)
    # return result_reg


def gen_stat(cg: Codegen, stat: ast.Stat):
    if isinstance(stat, ast.LocalVar):
        value_regs = []

        if stat.values:
            for i, v in enumerate(stat.values):
                if isinstance(v, ast.FunctionCall):
                    num_targets = (
                        len(stat.names) - i
                    )  # Handles multi return values when we're at the last value
                    reg_val = gen_function_call(cg, v, num_results=num_targets)
                else:
                    reg_val = gen_exp(cg, v)
                value_regs.append(reg_val)

        for name_token, reg_val in zip(stat.names, value_regs):
            cg.set_local_reg(name_token[0].lexeme, reg_val)

    elif isinstance(stat, ast.Assignment):
        value_regs = []

        for i, v in enumerate(stat.values):
            if isinstance(v, ast.FunctionCall):
                num_targets = (
                    len(stat.targets) - i
                )  # Handles multi return values when we're at the last value
                reg_val = gen_function_call(cg, v, num_results=num_targets)
            else:
                reg_val = gen_exp(cg, v)
            value_regs.append(reg_val)

        for target, reg_val in zip(stat.targets, value_regs):
            target_reg = cg.get_local_reg(target.name)
            cg.emit(Op.MOVE, a=target_reg, b=reg_val)

    elif isinstance(stat, ast.FunctionCall):
        gen_function_call(cg, stat)

    elif isinstance(stat, ast.While):
        cond_idx = len(cg.instructions)
        cond_reg = gen_exp(cg, stat.test)
        cg.emit(Op.TEST, a=cond_reg, c=0)
        jmp_idx = cg.emit(Op.JMP, a=0, sbx=0)
        gen_block(cg, stat.block)
        jmps_required = len(cg.instructions) - jmp_idx
        cg.emit(Op.JMP, a=0, sbx=cond_idx - len(cg.instructions) - 1)
        cg.instructions[jmp_idx] = Instruction(Op.JMP, a=0, sbx=jmps_required)

    elif isinstance(stat, ast.Repeat):
        body_idx = len(cg.instructions)
        print(body_idx)
        gen_block(cg, stat.block)
        cond_reg = gen_exp(cg, stat.test)
        cg.emit(Op.TEST, a=cond_reg, c=1)
        cg.emit(Op.JMP, a=0, sbx=1)
        cg.emit(Op.JMP, a=0, sbx=body_idx - len(cg.instructions) - 1)

    elif isinstance(stat, ast.If):
        end_jumps = []

        for i, branch in enumerate(stat.branches):
            cond, body = branch
            cond_reg = gen_exp(cg, cond)
            cg.emit(Op.TEST, a=cond_reg, c=0)
            cg.emit(Op.JMP, a=0, sbx=0)
            jmp_idx = len(cg.instructions) - 1
            gen_block(cg, body)
            jmps_required = len(cg.instructions) - jmp_idx

            end_jumps.append(len(cg.instructions))
            cg.emit(Op.JMP, a=0, sbx=0)

            cg.instructions[jmp_idx] = Instruction(Op.JMP, a=0, sbx=jmps_required)

        if stat.else_block:
            jmp_idx = len(cg.instructions)
            gen_block(cg, stat.else_block)

        for jmp_idx in end_jumps:
            jmps_required = len(cg.instructions) - jmp_idx - 1
            cg.instructions[jmp_idx] = Instruction(Op.JMP, a=0, sbx=jmps_required)

    else:
        print("unhandled stat:", stat)
        exit(1)


def gen_block(cg: Codegen, block: ast.Block):
    assert isinstance(block, ast.Block)

    for stat in block.stats:
        gen_stat(cg, stat)
