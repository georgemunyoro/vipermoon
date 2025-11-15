from typing import Optional


class UpValue:
    stack: Optional[list]
    index: Optional[int]

    def __init__(self, stack, index):
        self.open = True
        self.stack = stack
        self.index = index
        self.value = None

    def get(self):
        if self.open:
            assert self.stack is not None and self.index is not None
            return self.stack[self.index]
        return self.value

    def set(self, v):
        if self.open:
            assert self.stack is not None and self.index is not None
            self.stack[self.index] = v
        else:
            self.value = v

    def close(self):
        if self.open:
            assert self.stack is not None and self.index is not None
            self.value = self.stack[self.index]
            self.open = False
            self.stack = None
            self.index = None

    def __repr__(self):
        state = "open" if self.open else "closed"
        val = self.get()
        if self.open:
            location = f"stack[{self.index}]"
        else:
            location = "heap"
        return f"<UpValue {state} @ {location} = {val}>"
