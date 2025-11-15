from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


@dataclass
class Constant:
    class Kind(Enum):
        NIL = 0
        BOOLEAN = 1
        NUMBER = 3
        STRING = 4

    kind: Kind
    const: Optional[Union[str, int, float, bool]]

    def __repr__(self):
        return f"{self.kind} {self.const if self.kind != Constant.Kind.NIL else ''}"

    def __str__(self):
        match self.kind:
            case Constant.Kind.NIL:
                return "nil"
            case Constant.Kind.BOOLEAN:
                return str(self.const).lower()
            case Constant.Kind.STRING:
                return str(self.const)
            case Constant.Kind.NUMBER:
                return str(self.const)
