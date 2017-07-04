import collections

from fractions import Fraction
from multiprocessing import Pool
from threading import Semaphore


def input_semaphore_hook(g, s):
    for x in g:
        s.acquire()
        yield x


def output_semaphore_hook(g, s):
    for x in g:
        s.release()
        yield x


class ClopureFunction(object):
    def __init__(self, arg_vec, expression, name=""):
        self.name = name
        self.arg_vec = arg_vec
        self.expression = expression


class ClopureSymbol(object):
    def __init__(self, symbol):
        self.symbol = symbol

    def __repr__(self):
        return self.symbol

    def __eq__(self, v):
        if not isinstance(v, ClopureSymbol):
            return False
        return self.symbol == v.symbol


class ClopureRunner(object):
    def __init__(self, threads=1, queue_size=100):
        self.global_vars = {}

        self.threads = threads
        self.queue_size = queue_size

        self.core_functions = {
            "import": self.clopure_import,
            "defimport": self.clopure_defimport,
            "defimport-as": self.clopure_defimport_as,
            "def": self.clopure_def,
            "fn": self.clopure_fn,
            "defn": self.clopure_defn,
            "quote": self.clopure_quote,
            "eval": self.clopure_eval,
            "if": self.clopure_if,
            "do": self.clopure_do,
            "doseq": self.clopure_doseq,
            "map": self.clopure_map,
            "pmap": self.clopure_pmap,
            "pmap-unord": self.clopure_pmap_unord,
            "+": self.clopure_add,
            "-": self.clopure_sub,
            "*": self.clopure_mul,
            "/": self.clopure_div,
            "=": self.clopure_eq,
            ".": self.clopure_member,
        }


    def evaluate(self, node, local_vars={}):
        if isinstance(node, ClopureSymbol):
            symbol = node.symbol
            if symbol in self.global_vars:
                return self.evaluate(self.global_vars[symbol], local_vars=local_vars)
            if symbol in local_vars:
                return self.evaluate(local_vars[symbol], local_vars=local_vars)
            if symbol in self.core_functions:
                return self.core_functions[symbol]
            raise Exception("%s is not defined" % symbol)
        if isinstance(node, tuple):
            fn = self.evaluate(node[0], local_vars=local_vars)
            if isinstance(fn, ClopureFunction):
                new_local_vars = local_vars.copy()
                new_local_vars.update({k.symbol: self.evaluate(v, local_vars=local_vars)
                                                    for k, v in zip(fn.arg_vec, node[1:])})
                if fn.name:
                    new_local_vars[fn.name] = fn
                return self.evaluate(fn.expression, local_vars=new_local_vars)
            if fn in self.core_functions.values():
                return fn(*node[1:], local_vars=local_vars)
            if callable(fn):
                eval_args = [self.evaluate(arg, local_vars=local_vars) for arg in node[1:]]
                return fn(*eval_args)
            raise Exception("%s is not a function" % str(fn))
        if isinstance(node, list):
            return [self.evaluate(item, local_vars=local_vars) for item in node]
        return node


    def mp_evaluate_wrapper(self, x):
        return self.evaluate(x[0], local_vars=x[1])


    def clopure_import(self, *args, local_vars):
        for arg in args:
            if not isinstance(arg, ClopureSymbol):
                raise Exception("%s is not a symbol" % arg)
        if len(args) == 1:
            return __import__(args[0].symbol)
        elif len(args) == 2:
            return getattr(__import__(args[0].symbol, fromlist=[args[1].symbol]), args[1].symbol)
        else:
            raise Exception("import takes 1 or 2 arguments")


    def clopure_defimport(self, *args, local_vars):
        for arg in args:
            if not isinstance(arg, ClopureSymbol):
                raise Exception("%s is not a symbol" % arg)
        if len(args) == 1:
            self.global_vars[args[0].symbol] = __import__(args[0].symbol)
        elif len(args) == 2:
            self.global_vars[args[1].symbol] = getattr(__import__(args[0].symbol, fromlist=[args[1].symbol]), args[1].symbol)
        else:
            raise Exception("defimport takes 1 or 2 arguments")


    def clopure_defimport_as(self, *args, local_vars):
        for arg in args:
            if not isinstance(arg, ClopureSymbol):
                raise Exception("%s is not a symbol" % arg)
        if len(args) == 2:
            self.global_vars[args[0].symbol] = __import__(args[1].symbol)
        elif len(args) == 3:
            self.global_vars[args[0].symbol] = getattr(__import__(args[1].symbol, fromlist=[args[2].symbol]), args[2].symbol)
        else:
            raise Exception("defimport-as takes 2 or 3 arguments")


    def clopure_def(self, *args, local_vars):
        if len(args) != 2:
            raise Exception("def takes 2 arguments")
        if not isinstance(args[0], ClopureSymbol):
            raise Exception("%s is not a symbol" % str(args[0]))
        self.global_vars[args[0].symbol] = self.evaluate(args[1], local_vars=local_vars)


    def clopure_fn(self, *args, local_vars):
        if len(args) == 2:
            if not isinstance(args[0], list):
                raise Exception("the first argument must be a vector")
            for arg in args[0]:
                if not isinstance(arg, ClopureSymbol):
                    raise Exception("%s is not a symbol" % str(arg))
            return ClopureFunction(args[0], args[1])
        elif len(args) == 3:
            if not isinstance(args[0], ClopureSymbol):
                raise Exception("%s is not a symbol" % str(args[0]))
            if not isinstance(args[1], list):
                raise Exception("the second argument must be a vector")
            for arg in args[1]:
                if not isinstance(arg, ClopureSymbol):
                    raise Exception("%s is not a symbol" % str(arg))
            return ClopureFunction(args[1], args[2], name=args[0].symbol)
        raise Exception("fn takes 2 or 3 arguments")


    def clopure_defn(self, *args, local_vars):
        if len(args) != 3:
            raise Exception("fn takes 3 arguments")
        if not isinstance(args[0], ClopureSymbol):
            raise Exception("%s is not a symbol" % str(args[0]))
        if not isinstance(args[1], list):
            raise Exception("the first argument must be a vector")
        for arg in args[1]:
            if not isinstance(arg, ClopureSymbol):
                raise Exception("%s is not a symbol" % str(arg))
        self.global_vars[args[0].symbol] = ClopureFunction(args[1], args[2])


    def clopure_quote(self, *args, local_vars):
        if len(args) != 1:
            raise Exception("quote takes just 1 argument")
        return args[0]


    def clopure_eval(self, *args, local_vars):
        if len(args) != 1:
            raise Exception("eval takes just 1 argument")
        return self.evaluate(self.evaluate(args[0], local_vars=local_vars),
                        local_vars=local_vars)


    def clopure_if(self, *args, local_vars):
        if len(args) in (2, 3):
            r = self.evaluate(args[0], local_vars=local_vars)
            if r:
                return self.evaluate(args[1], local_vars=local_vars)
            if len(args) == 3:
                return self.evaluate(args[2], local_vars=local_vars)
            return None
        raise Exception("if takes 2 or 3 arguments")


    def clopure_do(self, *args, local_vars):
        ret = None
        for arg in args:
            ret = self.evaluate(arg, local_vars=local_vars)
        return ret


    def clopure_doseq(self, *args, local_vars):
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
                new_local_vars = local_vars.copy()
                new_local_vars.update({k.symbol: self.evaluate(v, local_vars=local_vars)
                                                    for k, v in zip(keys, chosen)})
                self.evaluate(func, local_vars=new_local_vars)
                return
            seq = self.evaluate(vals[0], local_vars=local_vars)
            for v in seq:
                doseq_loop(func, keys, vals[1:], chosen + [v])
        doseq_loop(args[1], keys, vals, [])


    def clopure_map(self, *args, local_vars):
        if len(args) <= 1:
            raise Exception("map takes at least 2 arguments")
        seqs = [self.evaluate(arg, local_vars=local_vars) for arg in args[1:]]
        return (self.evaluate((args[0],) + x, local_vars=local_vars) for x in zip(*seqs))


    def clopure_pmap(self, *args, local_vars):
        if len(args) <= 1:
            raise Exception("pmap takes at least 2 arguments")
        seqs = [self.evaluate(arg, local_vars=local_vars) for arg in args[1:]]
        p = Pool(self.threads)
        s = Semaphore(self.queue_size)
        input_iter = (((args[0],) + x, local_vars) for x in input_semaphore_hook(zip(*seqs), s))
        return output_semaphore_hook(p.imap(self.mp_evaluate_wrapper, input_iter), s)


    def clopure_pmap_unord(self, *args, local_vars):
        if len(args) <= 1:
            raise Exception("pmap-unord takes at least 2 arguments")
        seqs = [self.evaluate(arg, local_vars=local_vars) for arg in args[1:]]
        p = Pool(self.threads)
        s = Semaphore(self.queue_size)
        input_iter = (((args[0],) + x, local_vars) for x in input_semaphore_hook(zip(*seqs), s))
        return output_semaphore_hook(p.imap_unordered(self.mp_evaluate_wrapper, input_iter), s)


    def clopure_add(self, *args, local_vars):
        if len(args) == 0:
            raise Exception("* takes at least 1 argument")
        s = self.evaluate(args[0], local_vars=local_vars)
        for x in args[1:]:
            s += self.evaluate(x, local_vars=local_vars)
        return s


    def clopure_sub(self, *args, local_vars):
        if len(args) == 0:
            raise Exception("- takes at least 1 argument")
        s = self.evaluate(args[0], local_vars=local_vars)
        if len(args) == 1:
            return -s
        for x in args[1:]:
            s -= self.evaluate(x, local_vars=local_vars)
        return s


    def clopure_mul(self, *args, local_vars):
        if len(args) == 0:
            raise Exception("* takes at least 1 argument")
        s = self.evaluate(args[0], local_vars=local_vars)
        for x in args[1:]:
            s *= self.evaluate(x, local_vars=local_vars)
        return s


    def clopure_div(self, *args, local_vars):
        if len(args) == 0:
            raise Exception("/ takes at least 1 argument")
        s = self.evaluate(args[0], local_vars=local_vars)
        if len(args) == 1:
            return Fraction(1) / s
        if isinstance(s, int):
            s = Fraction(s)
        for x in args[1:]:
            s /= self.evaluate(x, local_vars=local_vars)
        return s


    def clopure_eq(self, *args, local_vars):
        if len(args) != 2:
            raise Exception("= takes 2 arguments")
        return self.evaluate(args[0], local_vars=local_vars) \
            == self.evaluate(args[1], local_vars=local_vars)


    def clopure_member(self, *args, local_vars):
        if len(args) == 0:
            raise Exception(". takes at least 1 argument")
        obj = self.evaluate(args[0], local_vars=local_vars)
        for arg in args[1:]:
            if not isinstance(arg, ClopureSymbol):
                raise Exception("%s is not a symbol" % str(arg))
            obj = getattr(obj, arg.symbol)
        return obj
