def lua_print(*args):
    strings = []
    for arg in args:
        if isinstance(arg, float) or isinstance(arg, int) or isinstance(arg, str):
            strings.append(str(arg))

        elif arg is None:
            strings.append("nil")

        else:
            raise Exception(f"unhandled print type: {arg}")

    print("\t".join(strings))


def create_globals():
    return {
        "print": lua_print,
    }
