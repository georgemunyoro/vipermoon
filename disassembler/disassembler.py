import struct
from dataclasses import dataclass
from enum import Enum
from io import BufferedReader
from pprint import pprint
from typing import List, Optional, Union


@dataclass
class Constant:
    class Kind(Enum):
        NIL = 0
        BOOLEAN = 1
        NUMBER = 3
        STRING = 4

    kind: Kind
    const: Optional[Union[str, int]]

    def __str__(self):
        return f"{self.kind} {self.const if self.kind != Constant.Kind.NIL else ''}"


class Instruction:
    def __init__(self, instr: int):
        self.instr = instr

    @property
    def op(self):
        return Op(self.instr & 0b111111)  # 6 bits

    @property
    def A(self):
        return (self.instr >> 6) & 0xFF  # 8 bits

    @property
    def C(self):
        return (self.instr >> 14) & 0x1FF  # 9 bits

    @property
    def B(self):
        return (self.instr >> 23) & 0x1FF  # 9 bits

    @property
    def Bx(self):
        return (self.instr >> 14) & 0x3FFFF  # 18 bits

    @property
    def sBx(self):
        return self.Bx - 131071  # 2^17 - 1

    def __str__(self):
        s = f"{self.op.name:<13} "

        def rk(arg: int):
            if arg >= 256:
                return -(arg - 255)
            return arg

        match self.op:
            case (
                Op.MOVE
                | Op.LOADNIL
                | Op.GETUPVAL
                | Op.SETUPVAL
                | Op.UNM
                | Op.NOT
                | Op.LEN
                | Op.RETURN
                | Op.VARARG
            ):
                s += f"{self.A} {self.B}"

            case (
                Op.LOADBOOL
                | Op.GETTABLE
                | Op.SETTABLE
                | Op.ADD
                | Op.SUB
                | Op.MUL
                | Op.DIV
                | Op.DIV
                | Op.POW
                | Op.CONCAT
                | Op.CALL
                | Op.TAILCALL
                | Op.SELF
                | Op.EQ
                | Op.LT
                | Op.LE
                | Op.TESTSET
                | Op.NEWTABLE
                | Op.SETLIST
            ):
                s += f"{self.A} {self.B} {rk(self.C)}"

            case Op.SETGLOBAL | Op.GETGLOBAL:
                s += f"{self.A} {-(self.Bx + 1)}"

            case Op.LOADK | Op.CLOSURE:
                s += f"{self.A} {self.Bx}"

            case Op.CLOSE:
                s += f"{self.A}"

            case Op.JMP:
                s += f"{self.sBx}"

            case Op.TEST | Op.TFORLOOP:
                s += f"{self.A} {self.C}"

            case Op.FORPREP | Op.FORLOOP:
                s += f"{self.A} {self.sBx}"

            case _:
                s += f"{self.A} {self.B} {self.C} {self.Bx} {self.sBx}"

        return s


@dataclass
class Local:
    name: str
    start: int
    end: int


@dataclass
class Prototype:
    source_name: str
    line_defined: int
    last_line_defined: int
    num_upvalues: int
    num_parameters: int
    is_vararg: int
    max_stack_size: int
    instructions: List[Instruction]
    constants: List[Constant]
    locals: List[Local]
    source_line_position_list: List[int]
    upvalues: List[str]
    prototypes: List["Prototype"]

    def dump(self):
        print(
            f"function <test.lua:{self.line_defined},{self.last_line_defined}> ({len(self.instructions)} instructions)"
        )
        print(
            f"{self.num_parameters} param, {len(self.locals) + len(self.constants) + len(self.prototypes) + len(self.upvalues)} slots, {len(self.upvalues)} upvalues, {len(self.locals)} local{'s' if len(self.locals) != 1 else ''}, {len(self.constants)} constants, {len(self.prototypes)} functions"
        )

        for i, instr in enumerate(self.instructions):
            print(
                f"\t{i+1:<7} {'[' + str(self.source_line_position_list[i]) + ']':<7} {instr}"
            )

        print(f"constants ({len(self.constants)}):")
        for i, const in enumerate(self.constants):
            print(f"\t{i+1:<7} {const.const}")

        print(f"locals ({len(self.locals)}):")
        for i, local in enumerate(self.locals):
            print(f"\t{i:<7} {local.name:<7} {local.start + 1:<7} {local.end + 1}")

        print(f"upvalues ({len(self.upvalues)}):")
        for i, upval in enumerate(self.upvalues):
            print(f"\t{i+1:<7} {upval}")

        print()

        for proto in self.prototypes:
            proto.dump()


class Op(Enum):
    MOVE = 0
    LOADK = 1
    LOADBOOL = 2
    LOADNIL = 3
    GETUPVAL = 4
    GETGLOBAL = 5
    GETTABLE = 6
    SETGLOBAL = 7
    SETUPVAL = 8
    SETTABLE = 9
    NEWTABLE = 10
    SELF = 11
    ADD = 12
    SUB = 13
    MUL = 14
    DIV = 15
    MOD = 16
    POW = 17
    UNM = 18
    NOT = 19
    LEN = 20
    CONCAT = 21
    JMP = 22
    EQ = 23
    LT = 24
    LE = 25
    TEST = 26
    TESTSET = 27
    CALL = 28
    TAILCALL = 29
    RETURN = 30
    FORLOOP = 31
    FORPREP = 32
    TFORLOOP = 33
    SETLIST = 34
    CLOSE = 35
    CLOSURE = 36
    VARARG = 37


@dataclass
class LuacHeader:
    version: int
    fmt: int
    endian: int
    int_size: int
    size_t_size: int
    instr_size: int
    num_size: int
    integral: int


def read_header(f: BufferedReader):
    sig = f.read(4)
    if sig != b"\x1bLua":
        raise Exception("Not a luac file")


class Disassembler:
    def __init__(self, bytecode: bytes):
        self.index = 0

        self.bytecode = bytecode

        assert self.read_byte() == 0x1B
        assert self.read_string(3) == "Lua"

        self.version = self.read_byte()
        self.version_hi = self.version >> 4
        self.version_lo = self.version & 0xF

        self.format = self.read_byte()
        self.endian = self.read_byte()
        self.int_size = self.read_byte()
        self.size_t_size = self.read_byte()
        self.instr_size = self.read_byte()
        self.num_size = self.read_byte()
        self.integral = self.read_byte()

    def print_header(self):
        print(
            f"lfile: Lua bytecode executable, version {self.version_hi}.{self.version_lo}"
        )
        print(f"   standard             {'yes' if self.format == 0 else 'no'}")
        print(
            f"   endianness           {'little' if self.endian == 1 else 'big' if self.endian == 0 else 'INVALID'}"
        )
        print(f"   sizeof(int)          {self.int_size}")
        print(f"   sizeof(instruction)  {self.instr_size}")
        print(f"   sizeof(size_t)       {self.size_t_size}")
        print(f"   sizeof(lua_Number)   {self.num_size}")
        print(
            f"   typeof(lua_Number)   {'float/double' if self.integral == 0 else 'unknown'}"
        )

        print()

        def warn(cond, msg):
            if cond:
                print("  ⚠️  " + msg)

        warn(self.endian not in (0, 1), "Invalid endianness (should be 0 or 1)")
        warn(self.int_size not in (4,), "Unusual int size (Lua expects 4 bytes)")
        warn(
            self.size_t_size not in (4, 8),
            "Unusual size_t size (4 or 8 bytes expected)",
        )
        warn(
            self.instr_size != 4, "Instruction size should be 4 bytes for standard Lua"
        )
        warn(
            self.num_size not in (4, 8),
            "Unusual lua_Number size (4 or 8 bytes expected)",
        )
        warn(self.integral not in (0, 1), "Integral flag must be 0/1")

    def read_bytes(self, size):
        b = self.bytecode[self.index : self.index + size]
        self.index += size
        return b

    def read_byte(self):
        byte = self.bytecode[self.index]
        self.index += 1
        return byte

    def read_string(self, size):
        string = "".join(chr(x) for x in self.bytecode[self.index : self.index + size])
        self.index += size
        return string

    def read_size_t(self):
        size_t = int.from_bytes(
            bytes=self.bytecode[self.index : self.index + self.size_t_size],
            byteorder="little" if self.endian else "big",
            signed=False,
        )
        self.index += self.size_t_size
        return size_t

    def read_int(self):
        int_ = int.from_bytes(
            bytes=self.bytecode[self.index : self.index + self.int_size],
            byteorder="little" if self.endian else "big",
            signed=False,
        )
        self.index += self.int_size
        return int_

    def read_number(self):
        f = struct.unpack(
            "<d" if self.endian else ">d", bytearray(self.read_bytes(self.num_size))
        )
        return f[0]

    def read_int32(self):
        f = struct.unpack("<d" if self.endian else ">d", bytearray(self.read_bytes(4)))
        return f[0]

    def read_prototype(self):
        proto = Prototype(
            source_name=self.read_string(self.read_size_t()),
            line_defined=self.read_int(),
            last_line_defined=self.read_int(),
            num_upvalues=self.read_byte(),
            num_parameters=self.read_byte(),
            is_vararg=self.read_byte(),
            max_stack_size=self.read_byte(),
            instructions=[],
            constants=[],
            locals=[],
            source_line_position_list=[],
            upvalues=[],
            prototypes=[],
        )

        num_instrs = self.read_int()
        for _ in range(num_instrs):
            proto.instructions.append(
                Instruction(
                    int.from_bytes(
                        bytes=self.bytecode[self.index : self.index + 4],
                        byteorder="little" if self.endian else "big",
                        signed=False,
                    )
                )
            )
            self.index += 4

        num_constants = self.read_int()
        for _ in range(num_constants):
            proto.constants.append(self.read_constant())

        sizep = self.read_int()
        for _ in range(sizep):
            proto.prototypes.append(self.read_prototype())

        sizelineinfo = self.read_int()
        for _ in range(sizelineinfo):
            proto.source_line_position_list.append(self.read_int())

        sizelocalvars = self.read_int()
        for _ in range(sizelocalvars):
            name = self.read_string(self.read_size_t())
            start = self.read_int()
            end = self.read_int()
            proto.locals.append(Local(name, start, end))

        sizeupvalues = self.read_int()
        for _ in range(sizeupvalues):
            proto.upvalues.append(self.read_string(self.read_int()))

        return proto

    def read_constant(self) -> Constant:
        kind = Constant.Kind(self.read_byte())
        if kind == Constant.Kind.NUMBER:
            value = self.read_number()
        elif kind == Constant.Kind.STRING:
            value = self.read_string(self.read_size_t())[:-1]
        elif kind == Constant.Kind.BOOLEAN:
            value = self.read_byte() == 1
        else:
            value = None
        return Constant(kind, value)

    def disassemble(self):
        return self.read_prototype()


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


def run_frame(frame: LuaFrame, gbl=create_gbl()):
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
        # print(f"{pc+1:<5} | {instr}")
        pc += 1

        match instr.op:
            case Op.MOVE:
                r[instr.A] = r[instr.B]

            case Op.LOADK:
                r[instr.A] = kst(instr.Bx)

            case Op.GETGLOBAL:
                r[instr.A] = gbl[kst(instr.Bx)]

            case Op.SETGLOBAL:
                gbl[kst(instr.Bx)] = r[instr.A]

            case Op.ADD:
                r[instr.A] = rk(instr.B) + rk(instr.C)
            case Op.SUB:
                r[instr.A] = rk(instr.B) - rk(instr.C)

            case Op.JMP:
                pc += instr.sBx

            case Op.LT:
                if (rk(instr.B) < rk(instr.C)) != instr.A:
                    pc += 1

            case Op.CALL:
                fn = r[instr.A]
                nargs = instr.B - 1
                args = stack[instr.A + 1 : instr.A + 1 + nargs]

                if callable(fn):
                    results = fn(*args)
                    if not isinstance(results, tuple):
                        results = (results,)

                else:
                    results = run_frame(LuaFrame(fn, args), gbl)

                nresults = instr.C - 1
                for i, val in enumerate(results[:nresults]):
                    stack[instr.A + i] = val

            case Op.RETURN:
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
                # TODO: handle upvalues
                stack[instr.A] = proto.prototypes[instr.Bx]

            case _:
                raise Exception(f"unhandled opcode: {instr.op}")
