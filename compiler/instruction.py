from enum import Enum


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


def iABx(opcode, A, Bx):
    return Instruction(opcode | ((A & 0xFF) << 6) | ((Bx & 0x3FFFF) << 14))


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
