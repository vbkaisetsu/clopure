import sys
import itertools


def clopure_add(*args):
    if len(args) == 0:
        return 0
    s = args[0]
    for x in args[1:]:
        s += x
    return s


def clopure_sub(*args):
    if len(args) == 0:
        raise ClopureRuntimeError("- takes at least 1 argument")
    s = args[0]
    if len(args) == 1:
        return -s
    for x in args[1:]:
        s -= x
    return s


def clopure_mul(*args):
    if len(args) == 0:
        return 1
    s = args[0]
    for x in args[1:]:
        s *= x
    return s


def clopure_div(*args):
    if len(args) == 0:
        raise ClopureRuntimeError("/ takes at least 1 argument")
    s = args[0]
    if len(args) == 1:
        return Fraction(1) / s
    if isinstance(s, int):
        s = Fraction(s)
    for x in args[1:]:
        s /= x
    return s


def clopure_mod(a, b):
    if len(args) != 2:
        raise ClopureRuntimeError("mod takes 2 arguments")
    return a % b


def clopure_eq(a, b):
    return a == b


def clopure_unzip(g, n):
    gs = itertools.tee(g, n)
    return [(lambda i: (x[i] for x in gs[i]))(i) for i in range(n)]


def clopure_arg_list(*args):
    return list(args)


def clopure_arg_tuple(*args):
    return tuple(args)


functions = {
    "+": clopure_add,
    "-": clopure_sub,
    "*": clopure_mul,
    "/": clopure_div,
    "mod": clopure_mod,
    "=": clopure_eq,
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "print": print,
    "read-line": sys.stdin.readline,
    "range": range,
    "reversed": reversed,
    "zip": lambda *x: (list(a) for a in zip(*x)),
    "unzip": clopure_unzip,
    "list": list,
    "tuple": tuple,
    "arg-list": clopure_arg_list,
    "arg-tuple": clopure_arg_tuple,
}
