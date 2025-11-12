from dataclasses import dataclass
from typing import Optional

from colorama import Fore, Style

from .location import Span


@dataclass
class CompileError(Exception):
    message: str
    span: Optional["Span"] = None
    filename: Optional[str] = None
    source_line: Optional[str] = None

    def __str__(self):
        loc = ""
        if self.span and self.span.start:
            loc = f"[line {self.span.start.line}, col {self.span.start.col}] "
        return f"{loc}{self.message}"


def format_error(err: CompileError) -> str:
    red = Fore.RED + Style.BRIGHT
    reset = Style.RESET_ALL
    parts = []
    if err.filename:
        parts.append(f"{Fore.CYAN}In {err.filename}:{reset}")

    if err.span and err.source_line:
        line = err.source_line.rstrip("\n")
        start = err.span.start.col - 1
        end = err.span.end.col - 1
        caret_line = " " * start + f"{red}{'^' * max(1, end - start)}{reset}"
        parts.append(line)
        parts.append(caret_line)

    parts.append(f"{red}{str(err)}{reset}")
    return "\n".join(parts)
