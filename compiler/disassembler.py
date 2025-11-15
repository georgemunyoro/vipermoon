import logging
import struct
from dataclasses import dataclass, field
from io import BufferedReader

from compiler.constant import Constant
from compiler.instruction import Instruction
from compiler.prototype import Local, Prototype

logging.basicConfig(
    level=logging.INFO,
)


@dataclass
class LuaTable:
    hash_size: int
    array_size: int
    fields: dict = field(default_factory=dict)

    def __setitem__(self, key, value):
        self.fields[key] = value

    def __getitem__(self, key):
        if value := self.fields.get(key):
            return value
        return Constant(Constant.Kind.NIL, None)


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
        logging.info(
            f"lfile: Lua bytecode executable, version {self.version_hi}.{self.version_lo}"
        )
        logging.info(f"   standard             {'yes' if self.format == 0 else 'no'}")
        logging.info(
            f"   endianness           {'little' if self.endian == 1 else 'big' if self.endian == 0 else 'INVALID'}"
        )
        logging.info(f"   sizeof(int)          {self.int_size}")
        logging.info(f"   sizeof(instruction)  {self.instr_size}")
        logging.info(f"   sizeof(size_t)       {self.size_t_size}")
        logging.info(f"   sizeof(lua_Number)   {self.num_size}")
        logging.info(
            f"   typeof(lua_Number)   {'float/double' if self.integral == 0 else 'unknown'}"
        )

        def warn(cond, msg):
            if cond:
                logging.warning(msg)

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
        if not self.endian:
            i = int.from_bytes(
                self.bytecode[self.index : self.index + 4],
                byteorder="big",
                signed=False,
            )
        else:
            i = int.from_bytes(
                self.bytecode[self.index : self.index + 4],
                byteorder="little",
                signed=False,
            )
        self.index = self.index + self.int_size
        return i

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
            start = self.read_int32()
            end = self.read_int32()
            proto.locals.append(Local(name, start, end))

        for _ in range(self.read_int()):
            proto.upvalues.append(self.read_string(self.read_size_t()))

        return proto

    def read_constant(self) -> Constant:
        kind = Constant.Kind(self.read_byte())
        if kind == Constant.Kind.NUMBER:
            value = self.read_number()
        elif kind == Constant.Kind.STRING:
            value = self.read_string(self.read_size_t())[:-1]
        elif kind == Constant.Kind.BOOLEAN:
            value = self.read_byte() != 0
        else:
            value = None
        return Constant(kind, value)

    def disassemble(self):
        return self.read_prototype()
