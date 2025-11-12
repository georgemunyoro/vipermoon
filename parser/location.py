from dataclasses import dataclass


@dataclass
class Location:
    index: int
    line: int
    col: int

    def __repr__(self):
        return f"{self.line}:{self.col}"


@dataclass
class Span:
    start: Location
    end: Location
