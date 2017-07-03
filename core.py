import collections

from fractions import Fraction


class ClopureFunction(object):
    def __init__(self, arg_vec, expression):
        self.arg_vec = arg_vec
        self.expression = expression


class ClopureSymbol(object):
    def __init__(self, symbol):
        self.symbol = symbol


def clopure_import(*args, global_vars, local_vars):
    for arg in args:
        if not isinstance(arg, ClopureSymbol):
            raise Exception("%s is not a symbol" % arg)
    if len(args) == 1:
        return __import__(args[0].symbol)
    elif len(args) == 2:
        return getattr(__import__(args[0].symbol, fromlist=[args[1].symbol]), args[1].symbol)
    else:
        raise Exception("import takes 1 or 2 arguments")


def clopure_defimport(*args, global_vars, local_vars):
    for arg in args:
        if not isinstance(arg, ClopureSymbol):
            raise Exception("%s is not a symbol" % arg)
    if len(args) == 2:
        global_vars[args[0].symbol] = __import__(args[1].symbol)
    elif len(args) == 3:
        global_vars[args[0].symbol] = getattr(__import__(args[1].symbol, fromlist=[args[2].symbol]), args[2].symbol)
    else:
        raise Exception("import takes 2 or 3 arguments")


def clopure_def(*args, global_vars, local_vars):
    if len(args) != 2:
        raise Exception("def takes 2 arguments")
    if not isinstance(args[0], ClopureSymbol):
        raise Exception("%s is not a symbol" % str(args[0]))
    global_vars[args[0].symbol] = args[1]


def clopure_fn(*args, global_vars, local_vars):
    if len(args) != 2:
        raise Exception("fn takes 2 variables")
    if not isinstance(args[0], list):
        raise Exception("the first argument must be a vector")
    for arg in args[0]:
        if not isinstance(arg, ClopureSymbol):
            raise Exception("%s is not a symbol" % str(arg))
    return ClopureFunction(args[0], args[1])


def clopure_defn(*args, global_vars, local_vars):
    if len(args) != 3:
        raise Exception("fn takes 3 variables")
    if not isinstance(args[0], ClopureSymbol):
        raise Exception("%s is not a symbol" % str(args[0]))
    if not isinstance(args[1], list):
        raise Exception("the first argument must be a vector")
    for arg in args[1]:
        if not isinstance(arg, ClopureSymbol):
            raise Exception("%s is not a symbol" % str(arg))
    global_vars[args[0].symbol] = ClopureFunction(args[1], args[2])


def clopure_if(*args, global_vars, local_vars):
    if len(args) in (2, 3):
        r = evaluate(args[0], global_vars=global_vars, local_vars=local_vars)
        if r:
            return evaluate(args[1], global_vars=global_vars, local_vars=local_vars)
        if len(args) == 3:
            return evaluate(args[2], global_vars=global_vars, local_vars=local_vars)
        return None
    raise Exception("if takes 2 or 3 arguments")


def clopure_do(*args, global_vars, local_vars):
    ret = None
    for arg in args:
        ret = evaluate(arg, global_vars=global_vars, local_vars=local_vars)
    return ret


def clopure_doseq(*args, global_vars, local_vars):
    if len(args) != 2:
        raise Exception("doesq takes 2 arguments")
    if not isinstance(args[0], list):
        raise Exception("the second argument must be a vector")
    if len(args[0]) % 2 != 0:
        raise Exception("the second argument must contain even number of forms")
    keys = []
    vals = []
    for i in range(0, len(args[0]), 2):
        if not isinstance(args[0][i], ClopureSymbol):
            raise Exception("%s is not a symbol" % str(args[0][i]))
        keys.append(args[0][i])
        vals.append(args[0][i + 1])
    def doseq_loop(func, keys, vals, chosen):
        if not vals:
            new_local_vars = {k.symbol: v for k, v in zip(keys, chosen)}
            new_local_vars.update(local_vars)
            evaluate(func, global_vars=global_vars, local_vars=new_local_vars)
            return
        seq = evaluate(vals[0], global_vars=global_vars, local_vars=local_vars)
        for v in seq:
            doseq_loop(func, keys, vals[1:], chosen + [v])
    doseq_loop(args[1], keys, vals, [])


def clopure_add(*args, global_vars, local_vars):
    if len(args) == 0:
        raise Exception("* requires at least 1 arguments")
    s = evaluate(args[0], global_vars=global_vars, local_vars=local_vars)
    for x in args[1:]:
        s += evaluate(x, global_vars=global_vars, local_vars=local_vars)
    return s


def clopure_sub(*args, global_vars, local_vars):
    if len(args) == 0:
        raise Exception("- requires at least 1 arguments")
    s = evaluate(args[0], global_vars=global_vars, local_vars=local_vars)
    if len(args) == 1:
        return -s
    for x in args[1:]:
        s -= evaluate(x, global_vars=global_vars, local_vars=local_vars)
    return s


def clopure_mul(*args, global_vars, local_vars):
    if len(args) == 0:
        raise Exception("* requires at least 1 arguments")
    s = evaluate(args[0], global_vars=global_vars, local_vars=local_vars)
    for x in args[1:]:
        s *= evaluate(x, global_vars=global_vars, local_vars=local_vars)
    return s


def clopure_div(*args, global_vars, local_vars):
    if len(args) == 0:
        raise Exception("/ requires at least 1 arguments")
    s = evaluate(args[0], global_vars=global_vars, local_vars=local_vars)
    if len(args) == 1:
        return Fraction(1) / s
    if isinstance(s, int):
        s = Fraction(s)
    for x in args[1:]:
        s /= evaluate(x, global_vars=global_vars, local_vars=local_vars)
    return s


def clopure_eq(*args, global_vars, local_vars):
    if len(args) != 2:
        raise Exception("= requires 2 arguments")
    return evaluate(args[0], global_vars=global_vars, local_vars=local_vars) \
        == evaluate(args[1], global_vars=global_vars, local_vars=local_vars)


core_functions = {
    "import": clopure_import,
    "defimport": clopure_defimport,
    "def": clopure_def,
    "fn": clopure_fn,
    "defn": clopure_defn,
    "if": clopure_if,
    "do": clopure_do,
    "doseq": clopure_doseq,
    "+": clopure_add,
    "-": clopure_sub,
    "*": clopure_mul,
    "/": clopure_div,
    "=": clopure_eq,
}


def evaluate(node, global_vars, local_vars):
    if isinstance(node, ClopureSymbol):
        symbol = node.symbol
        if symbol in global_vars:
            return evaluate(global_vars[symbol], global_vars=global_vars, local_vars=local_vars)
        if symbol in local_vars:
            return evaluate(local_vars[symbol], global_vars=global_vars, local_vars=local_vars)
        if symbol in core_functions:
            return core_functions[symbol]
        raise Exception("%s is not defined" % symbol)
    if isinstance(node, tuple):
        fn = evaluate(node[0], global_vars=global_vars, local_vars=local_vars)
        if isinstance(fn, ClopureFunction):
            new_local_vars = {k.symbol: v for k, v in zip(fn.arg_vec, node[1:])}
            new_local_vars.update(local_vars)
            return evaluate(fn.expression, global_vars=global_vars, local_vars=new_local_vars)
        if fn in core_functions.values():
            return fn(*node[1:], global_vars=global_vars, local_vars=local_vars)
        if callable(fn):
            eval_args = [evaluate(arg, global_vars=global_vars, local_vars=local_vars) for arg in node[1:]]
            return fn(*eval_args)
        raise Exception("%s is not a function" % str(fn))
    return node
