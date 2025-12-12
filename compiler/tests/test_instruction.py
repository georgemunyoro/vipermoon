from compiler.instruction import Instruction, Op


def test_instruction_op():
    for op in Op:
        instr = Instruction(op, a=0)
        assert instr.op == op


def test_instruction_a():
    for a in range(256):
        instr = Instruction(Op.MOVE, a=a)
        assert instr.A == a


def test_instruction_b():
    for b in range(512):
        instr = Instruction(Op.MOVE, a=0, b=b)
        assert instr.B == b


def test_instruction_c():
    for c in range(512):
        instr = Instruction(Op.MOVE, a=0, c=c)
        assert instr.C == c


def test_instruction_bx():
    for bx in range(0x3FFFF):
        instr = Instruction(Op.MOVE, a=0, bx=bx)
        assert instr.Bx == bx


def test_instruction_sbx():
    for sbx in range(-131071, 131071):
        instr = Instruction(Op.MOVE, a=0, sbx=sbx)
        assert instr.sBx == sbx
