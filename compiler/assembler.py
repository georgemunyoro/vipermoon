import struct

from .constant import Constant
from .prototype import Prototype


class Assembler:
    def __init__(self, outfile="vipermoon.out"):
        self.bytecode = bytearray()
        self.outfile = outfile

    def save(self):
        with open(self.outfile, "wb") as f:
            f.write(self.bytecode)

    def write_header(self):
        self.write_byte(0x1B)
        self.write_string("Lua", False)
        self.write_byte(0x51)  # version
        self.write_byte(0)  # format
        self.write_byte(1)  # endianness
        self.write_byte(4)  # int size
        self.write_byte(8)  # size_t size
        self.write_byte(4)  # instr size
        self.write_byte(8)  # num size
        self.write_byte(0)  # integral flag

    def write_byte(self, val):
        self.bytecode.append(val)

    def write_string(self, s: str, with_terminator=True):
        for char in s:
            self.bytecode.append(ord(char))
        if with_terminator:
            self.bytecode.append(0)

    def write_size_t(self, val):
        size_t = struct.pack("<Q", val)
        for b in size_t:
            self.bytecode.append(b)

    def write_int(self, val):
        i = struct.pack("<I", val)
        assert len(i) == 4
        for b in i:
            self.bytecode.append(b)

    def write_number(self, val):
        i = struct.pack("<d", val)
        assert len(i) == 8
        for b in i:
            self.bytecode.append(b)

    def write_proto(self, proto: Prototype):
        self.write_size_t(len(proto.source_name) + 2)
        self.write_string(f"@{proto.source_name}")
        self.write_int(proto.line_defined)
        self.write_int(proto.last_line_defined)
        self.write_byte(len(proto.upvalues))
        self.write_byte(proto.num_parameters)
        self.write_byte(2)
        self.write_byte(proto.max_stack_size)

        self.write_int(len(proto.instructions))
        for instr in proto.instructions:
            for b in struct.pack("I", instr.instr):
                self.bytecode.append(b)

        self.write_int(len(proto.constants))
        for const in proto.constants:
            self.write_byte(const.kind.value)
            if const.kind == Constant.Kind.NUMBER:
                self.write_number(const.const)
            elif const.kind == Constant.Kind.STRING:
                self.write_size_t(len(str(const.const)) + 1)
                self.write_string(str(const.const))
            elif const.kind == Constant.Kind.BOOLEAN:
                self.write_byte(bool(const.const))

        self.write_int(len(proto.prototypes))
        for p in proto.prototypes:
            self.write_proto(p)

        self.write_int(0)  # TODO: source line position list

        self.write_int(len(proto.locals))
        for local in proto.locals:
            self.write_size_t(len(local.name) + 1)
            self.write_string(local.name)
            self.write_int(local.start)
            self.write_int(local.end)

        self.write_int(0)  # TODO: upvalues list
