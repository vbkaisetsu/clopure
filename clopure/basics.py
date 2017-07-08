import sys
import itertools
import operator

from functools import reduce
from fractions import Fraction


def clopure_div(s, *args):
    s = Fraction(s) if isinstance(s, int) else s
    if len(args) == 0:
        return Fraction(1) / s
    for x in args:
        s /= x
    return s


def clopure_unzip(g, n):
    gs = itertools.tee(g, n)
    return [(lambda i: (x[i] for x in gs[i]))(i) for i in range(n)]


functions = {
    "+": lambda *x: sum(x),
    "-": lambda s, *x: reduce(operator.sub, x, s) if len(x) != 0 else -s,
    "*": lambda *x: reduce(operator.mul, x, 1),
    "/": clopure_div,
    "mod": lambda a, b: a % b,
    "=": lambda a, b: a == b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "not": lambda x: not x,
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "print": print,
    "input": input,
    "read-line": sys.stdin.readline,
    "range": range,
    "reversed": reversed,
    "zip": lambda *x: (list(a) for a in zip(*x)),
    "unzip": clopure_unzip,
    "enumerate": lambda x: (list(a) for a in enumerate(x)),
    "list": list,
    "tuple": tuple,
    "arg-list": lambda *x: list(x),
    "arg-tuple": lambda *x: tuple(x),
}
