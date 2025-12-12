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

    def emit(self, op: Op, a: int, b=None, c=None, bx=None, sbx=None):
        assert a is not None
        instr = Instruction(op, a, b, c, bx, sbx)
        print(instr)
        self.instructions.append(instr)

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

        opcode = None
        match exp.op.kind:
            case TokenKind.PLUS:
                opcode = Op.ADD
            case TokenKind.STAR:
                opcode = Op.MUL
            case TokenKind.MINUS:
                opcode = Op.SUB
            case TokenKind.MOD:
                opcode = Op.MOD
            case TokenKind.SLASH:
                opcode = Op.DIV
            case TokenKind.POW:
                opcode = Op.POW

        if not opcode:
            print(f"invalid binary op: {exp.op.lexeme}")
            exit(1)

        dest = cg.alloc_reg()
        print("reg", cg.reg_count)

        cg.emit(opcode, a=dest, b=left, c=right)
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

    else:
        print("unhandled stat:", stat)
        exit(1)


def gen_block(cg: Codegen, block: ast.Block):
    assert isinstance(block, ast.Block)

    for stat in block.stats:
        gen_stat(cg, stat)
