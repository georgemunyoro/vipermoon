import logging
from dataclasses import dataclass
from pprint import pformat
from typing import List

from compiler.constant import Constant
from compiler.disassembler import LuaTable
from compiler.instruction import Op
from compiler.prototype import Prototype
from interpreter.upvalue import UpValue


class LuaFrame:
    def __init__(self, proto: Prototype, args=[]):
        self.proto = proto
        self.stack: list = [None] * proto.max_stack_size
        for i, arg in enumerate(args):
            self.stack[i] = arg

        self.pc = 0
        self.upvalues = [None] * len(proto.upvalues)


def create_gbl():
    gbl = dict()

    gbl["print"] = lambda *args: print(*args)

    return gbl


@dataclass
class Closure:
    proto: Prototype
    upvalues: List[UpValue]


def is_truthy(arg):
    if isinstance(arg, int) or isinstance(arg, float):
        return arg != 0
    elif isinstance(arg, bool):
        return arg
    elif isinstance(arg, Constant):
        if arg.kind == Constant.Kind.NIL:
            return False
    raise Exception(f"unhandled type: {arg}")


def run_frame(frame: LuaFrame, gbl=create_gbl(), upvalues: List[UpValue] = []):
    proto = frame.proto
    stack = frame.stack
    r = frame.stack
    pc = frame.pc

    def kst(idx):
        return proto.constants[idx].const

    def loop_continues(index, limit, step):
        return index <= limit if step > 0 else index >= limit

    def rk(arg):
        if arg >= 256:
            return proto.constants[arg - 256].const
        return stack[arg]

    while pc < len(proto.instructions):
        instr = proto.instructions[pc]
        # logging.debug(f"{pc+1:<5} | {instr}")
        pc += 1

        match instr.op:
            case Op.MOVE:
                r[instr.A] = r[instr.B]

            case Op.LOADNIL:
                for i in range(instr.A, instr.B + 1):
                    r[i] = Constant(Constant.Kind.NIL, None)

            case Op.GETUPVAL:
                upvalue = upvalues[instr.B]
                logging.debug(f"getting upvalue with id={id(upvalue)}")
                r[instr.A] = upvalue.get()

            case Op.LOADK:
                r[instr.A] = kst(instr.Bx)

            case Op.LOADBOOL:
                r[instr.A] = is_truthy(instr.B)
                if bool(instr.C):
                    pc += 1

            case Op.GETGLOBAL:
                r[instr.A] = gbl[kst(instr.Bx)]

            case Op.GETTABLE:
                table = r[instr.A]
                key = rk(instr.C)
                logging.debug(
                    f"getting key={key} of table with id={id(table)}, internal dict id={id(table.fields)}"
                )
                r[instr.A] = table[key]

            case Op.SETGLOBAL:
                gbl[kst(instr.Bx)] = r[instr.A]

            case Op.SETUPVAL:
                upvalue = upvalues[instr.B]
                logging.debug(f"setting upvalue with id={id(upvalue)}")
                upvalue.set(r[instr.A])

            case Op.SETTABLE:
                table = r[instr.A]
                key = rk(instr.B)
                val = rk(instr.C)

                logging.debug(pformat(table.fields))

                logging.debug(
                    f"setting key={key} of table with id={id(table)}, internal dict id={id(table.fields)} to value={val}"
                )
                logging.debug(f"SETTABLE BEFORE : {pformat(table.fields)}")
                table[key] = val
                logging.debug(f"SETTABLE AFTER  : {pformat(table.fields)}")

            case Op.NEWTABLE:
                table = LuaTable(instr.B, instr.C)
                logging.debug(
                    f"created lua table with id={id(table)}, internal dict id={id(table.fields)}"
                )
                r[instr.A] = table

            case Op.ADD | Op.SUB:
                operators = {}
                operators[Op.ADD.name] = lambda a, b: a + b
                operators[Op.SUB.name] = lambda a, b: a - b

                r[instr.A] = operators[instr.op.name](rk(instr.B), rk(instr.C))

            case Op.CONCAT:
                assert instr.C > instr.B
                r[instr.A] = "".join([str(v) for v in r[instr.B : instr.C + 1]])

            case Op.JMP:
                pc += instr.sBx

            case Op.LT | Op.EQ | Op.LE:
                operators = {}
                operators[Op.LT.name] = lambda a, b: a < b
                operators[Op.LE.name] = lambda a, b: a <= b
                operators[Op.EQ.name] = lambda a, b: a == b

                if (operators[instr.op.name](rk(instr.B), rk(instr.C))) != instr.A:
                    pc += 1

            case Op.TEST:
                if not is_truthy(r[instr.A]) == instr.C:
                    pc += 1

            case Op.TESTSET:
                if is_truthy(r[instr.B]) == instr.C:
                    r[instr.A] = r[instr.B]
                else:
                    pc += 1

            case Op.CALL:
                if instr.B == 0:
                    args = stack[instr.A + 1 :]
                else:
                    args = stack[instr.A + 1 : instr.A + 1 + instr.B - 1]

                closure: Union[Callable, Closure] = r[instr.A]

                if callable(closure):
                    results = closure(*args)
                    if not isinstance(results, tuple):
                        results = (results,)

                else:
                    results = run_frame(
                        LuaFrame(closure.proto, args),
                        gbl,
                        closure.upvalues,
                    )

                if instr.C == 0:
                    nresults = len(results)
                else:
                    nresults = instr.C - 1

                for i, val in enumerate(results[:nresults]):
                    stack[instr.A + i] = val

            case Op.CLOSE:
                for uv in upvalues:
                    if uv.open:
                        assert uv.index is not None
                        if uv.index >= instr.A:
                            uv.close()

            case Op.RETURN:
                for uv in upvalues:
                    if uv.open:
                        assert uv.index is not None
                        if uv.index >= instr.A:
                            uv.close()

                if instr.B == 1:
                    return []
                elif instr.B == 0:
                    return stack[instr.A :]
                else:
                    return stack[instr.A : instr.A + instr.B - 1]

            case Op.FORLOOP:
                r[instr.A] += r[instr.A + 2]

                if loop_continues(r[instr.A], r[instr.A + 1], r[instr.A + 2]):
                    pc += instr.sBx
                    r[instr.A + 3] = r[instr.A]

            case Op.FORPREP:
                r[instr.A] -= r[instr.A + 2]
                pc += instr.sBx

            case Op.CLOSURE:
                closure = Closure(proto.prototypes[instr.Bx], [])
                for i, uv in enumerate(closure.proto.upvalues):
                    pseudo_instr = proto.instructions[pc + i]
                    if pseudo_instr.op == Op.MOVE:
                        closure.upvalues.append(UpValue(stack, pseudo_instr.B))
                    elif pseudo_instr.op == Op.GETUPVAL:
                        closure.upvalues.append(upvalues[pseudo_instr.B])
                    else:
                        raise Exception(
                            f"Unexpected pseudo-instruction after CLOSURE: {pseudo_instr.op}"
                        )
                r[instr.A] = closure
                logging.debug(f"created closure: {pformat(closure)}")
                for i, uv in enumerate(closure.upvalues):
                    logging.debug(f" {i} {uv} {id(uv)} {id(uv.get())}")
                pc += len(closure.proto.upvalues)

            case _:
                raise Exception(f"unhandled opcode: {instr.op}")

        msg = pformat({"pc": pc, "stack": frame.stack, "upvalues": upvalues})
        logging.debug(msg)
