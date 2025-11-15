import logging
from dataclasses import dataclass
from typing import List

from compiler.constant import Constant
from compiler.instruction import Instruction


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
        logging.info(
            f"function <test.lua:{self.line_defined},{self.last_line_defined}> ({len(self.instructions)} instructions)"
        )
        logging.info(
            f"{self.num_parameters} param, {len(self.locals) + len(self.constants) + len(self.prototypes) + len(self.upvalues)} slots, {len(self.upvalues)} upvalues, {len(self.locals)} local{'s' if len(self.locals) != 1 else ''}, {len(self.constants)} constants, {len(self.prototypes)} functions"
        )

        for i, instr in enumerate(self.instructions):
            logging.info(
                f"\t{i+1:<7} {'[' + str(self.source_line_position_list[i]) + ']':<7} {instr}"
            )

        logging.info(f"constants ({len(self.constants)}):")
        for i, const in enumerate(self.constants):
            logging.info(f"\t{i+1:<7} {const.const}")

        logging.info(f"locals ({len(self.locals)}):")
        for i, local in enumerate(self.locals):
            logging.info(
                f"\t{i:<7} {local.name:<7} {local.start + 1:<7} {local.end + 1}"
            )

        logging.info(f"upvalues ({len(self.upvalues)}):")
        for i, upval in enumerate(self.upvalues):
            logging.info(f"\t{i+1:<7} {upval}")

        for proto in self.prototypes:
            proto.dump()

    def __repr__(self):
        return (
            f"<Prototype {self.source_name or '<anonymous>'} "
            f"lines {self.line_defined}-{self.last_line_defined}, "
            f"upvalues={self.num_upvalues}, params={self.num_parameters}, "
            f"vararg={self.is_vararg}, max_stack={self.max_stack_size}, "
            f"instructions={len(self.instructions)}, "
            f"constants={len(self.constants)}, locals={len(self.locals)}, "
            f"prototypes={len(self.prototypes)}>"
        )
