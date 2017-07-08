import collections
import itertools
import traceback
import sys

from multiprocessing import Pool, Semaphore, Process, Pipe, Queue, Lock
from threading import Thread

from clopure import extras, basics
from clopure.exceptions import ClopureRuntimeError


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


class EOFMessage(object):
    pass


class ReducedObject(object):
    def __init__(self, value):
        self.value = value


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
            "chain-args": self.clopure_chain_args,
            "quote": self.clopure_quote,
            "eval": self.clopure_eval,
            "if": self.clopure_if,
            "do": self.clopure_do,
            "doseq": self.clopure_doseq,
            "dorun": self.clopure_dorun,
            "map": self.clopure_map,
            "reduce": self.clopure_reduce,
            "reduced": self.clopure_reduced,
            "filter": self.clopure_filter,
            "remove": self.clopure_remove,
            "take-while": self.clopure_take_while,
            "pmap": self.clopure_pmap,
            "pmap-unord": self.clopure_pmap_unord,
            "iter-mp-split": self.clopure_iter_mp_split,
            "iter-mp-split-unord": self.clopure_iter_mp_split_unord,
            "and": self.clopure_and,
            "or": self.clopure_or,
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
            if symbol in basics.functions:
                return basics.functions[symbol]
            if symbol in extras.functions:
                return extras.functions[symbol]
            raise ClopureRuntimeError("%s is not defined" % symbol)
        if isinstance(node, tuple):
            fn = self.evaluate(node[0], local_vars=local_vars)
            if isinstance(fn, ClopureFunction):
                new_local_vars = local_vars.copy()
                new_local_vars.update({k.symbol: v for k, v in zip(fn.arg_vec, node[1:])})
                if fn.name:
                    new_local_vars[fn.name] = fn
                return self.evaluate(fn.expression, local_vars=new_local_vars)
            if fn in self.core_functions.values():
                return fn(*node[1:], local_vars=local_vars)
            if callable(fn):
                eval_args = [self.evaluate(arg, local_vars=local_vars) for arg in node[1:]]
                return fn(*eval_args)
            raise ClopureRuntimeError("%s is not a function" % str(fn))
        if isinstance(node, list):
            return [self.evaluate(item, local_vars=local_vars) for item in node]
        return node


    def mp_evaluate_wrapper(self, x):
        return self.evaluate(x[0], local_vars=x[1])


    def iter_split_evaluate_wrapper(self, fn, local_vars, in_size, q_in, q_out):
        l = Lock()
        idx_q = Queue()
        def split_iter():
            try:
                while True:
                    l.acquire()
                    i, data_in = q_in.get()
                    idx_q.put(i)
                    if data_in is EOFMessage:
                        return
                    yield data_in
            except BaseException:
                traceback.print_exc(file=sys.stdout)
        gs = itertools.tee(split_iter(), in_size)
        for data_out in self.evaluate((fn,) + tuple((lambda i: (x[i] for x in gs[i]))(i) for i in range(in_size)), local_vars=local_vars):
            q_out.put((idx_q.get(), data_out))
            l.release()
        q_out.put((0, EOFMessage))


    def clopure_import(self, *args, local_vars):
        for arg in args:
            if not isinstance(arg, ClopureSymbol):
                raise ClopureRuntimeError("%s is not a symbol" % arg)
        if len(args) == 1:
            return __import__(args[0].symbol)
        elif len(args) == 2:
            return getattr(__import__(args[0].symbol, fromlist=[args[1].symbol]), args[1].symbol)
        else:
            raise ClopureRuntimeError("import takes 1 or 2 arguments")


    def clopure_defimport(self, *args, local_vars):
        for arg in args:
            if not isinstance(arg, ClopureSymbol):
                raise ClopureRuntimeError("%s is not a symbol" % arg)
        if len(args) == 1:
            self.global_vars[args[0].symbol] = __import__(args[0].symbol)
        elif len(args) == 2:
            self.global_vars[args[1].symbol] = getattr(__import__(args[0].symbol, fromlist=[args[1].symbol]), args[1].symbol)
        else:
            raise ClopureRuntimeError("defimport takes 1 or 2 arguments")


    def clopure_defimport_as(self, *args, local_vars):
        for arg in args:
            if not isinstance(arg, ClopureSymbol):
                raise ClopureRuntimeError("%s is not a symbol" % arg)
        if len(args) == 2:
            self.global_vars[args[0].symbol] = __import__(args[1].symbol)
        elif len(args) == 3:
            self.global_vars[args[0].symbol] = getattr(__import__(args[1].symbol, fromlist=[args[2].symbol]), args[2].symbol)
        else:
            raise ClopureRuntimeError("defimport-as takes 2 or 3 arguments")


    def clopure_def(self, *args, local_vars):
        if len(args) != 2:
            raise ClopureRuntimeError("def takes 2 arguments")
        if not isinstance(args[0], ClopureSymbol):
            raise ClopureRuntimeError("%s is not a symbol" % str(args[0]))
        self.global_vars[args[0].symbol] = self.evaluate(args[1], local_vars=local_vars)


    def clopure_fn(self, *args, local_vars):
        if len(args) == 2:
            if not isinstance(args[0], list):
                raise ClopureRuntimeError("the first argument must be a vector")
            for arg in args[0]:
                if not isinstance(arg, ClopureSymbol):
                    raise ClopureRuntimeError("%s is not a symbol" % str(arg))
            return ClopureFunction(args[0], args[1])
        elif len(args) == 3:
            if not isinstance(args[0], ClopureSymbol):
                raise ClopureRuntimeError("%s is not a symbol" % str(args[0]))
            if not isinstance(args[1], list):
                raise ClopureRuntimeError("the second argument must be a vector")
            for arg in args[1]:
                if not isinstance(arg, ClopureSymbol):
                    raise ClopureRuntimeError("%s is not a symbol" % str(arg))
            return ClopureFunction(args[1], args[2], name=args[0].symbol)
        raise ClopureRuntimeError("fn takes 2 or 3 arguments")


    def clopure_defn(self, *args, local_vars):
        if len(args) != 3:
            raise ClopureRuntimeError("fn takes 3 arguments")
        if not isinstance(args[0], ClopureSymbol):
            raise ClopureRuntimeError("%s is not a symbol" % str(args[0]))
        if not isinstance(args[1], list):
            raise ClopureRuntimeError("the first argument must be a vector")
        for arg in args[1]:
            if not isinstance(arg, ClopureSymbol):
                raise ClopureRuntimeError("%s is not a symbol" % str(arg))
        self.global_vars[args[0].symbol] = ClopureFunction(args[1], args[2])


    def clopure_chain_args(self, *args, local_vars):
        if len(args) == 0:
            raise ClopureRuntimeError("chain-args takes 1 or more arguments")
        seqs = [self.evaluate(arg, local_vars=local_vars) for arg in args[1:]]
        return self.evaluate((args[0],) + tuple(itertools.chain(*seqs)), local_vars=local_vars)


    def clopure_quote(self, *args, local_vars):
        if len(args) != 1:
            raise ClopureRuntimeError("quote takes just 1 argument")
        return args[0]


    def clopure_eval(self, *args, local_vars):
        if len(args) != 1:
            raise ClopureRuntimeError("eval takes just 1 argument")
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
        raise ClopureRuntimeError("if takes 2 or 3 arguments")


    def clopure_do(self, *args, local_vars):
        ret = None
        for arg in args:
            ret = self.evaluate(arg, local_vars=local_vars)
        return ret


    def clopure_doseq(self, *args, local_vars):
        if len(args) != 2:
            raise ClopureRuntimeError("doesq takes 2 arguments")
        if not isinstance(args[0], list):
            raise ClopureRuntimeError("the second argument must be a vector")
        if len(args[0]) % 2 != 0:
            raise ClopureRuntimeError("the second argument must contain even number of forms")
        keys = []
        vals = []
        for i in range(0, len(args[0]), 2):
            if not isinstance(args[0][i], ClopureSymbol):
                raise ClopureRuntimeError("%s is not a symbol" % str(args[0][i]))
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


    def clopure_dorun(self, *args, local_vars):
        if len(args) != 1:
            raise ClopureRuntimeError("dorun takes 1 argument")
        for item in self.evaluate(args[0], local_vars=local_vars):
            self.evaluate(item, local_vars=local_vars)


    def clopure_map(self, *args, local_vars):
        if len(args) <= 1:
            raise ClopureRuntimeError("map takes at least 2 arguments")
        seqs = [self.evaluate(arg, local_vars=local_vars) for arg in args[1:]]
        return (self.evaluate((args[0],) + x, local_vars=local_vars) for x in zip(*seqs))


    def clopure_reduce(self, *args, local_vars):
        reduce_args = []
        if len(args) == 3:
            reduce_args.append(args[1])
            g = self.evaluate(args[2], local_vars=local_vars)
        elif len(args) == 2:
            g = self.evaluate(args[1], local_vars=local_vars)
        else:
            raise ClopureRuntimeError("reduce takes 2 or 3 arguments")
        for item in g:
            if len(reduce_args) == 2:
                result = self.evaluate((args[0],) + tuple(reduce_args), local_vars=local_vars)
                if isinstance(result, ReducedObject):
                    return result.value
                reduce_args = [result]
            reduce_args.append(item)
        result = self.evaluate((args[0],) + tuple(reduce_args), local_vars=local_vars)
        if isinstance(result, ReducedObject):
            return result.value
        return result


    def clopure_reduced(self, *args, local_vars):
        if len(args) != 1:
            raise ClopureRuntimeError("reduced takes 1 argument")
        return ReducedObject(args[0])


    def clopure_filter(self, *args, local_vars):
        if len(args) != 2:
            raise ClopureRuntimeError("filter takes 2 arguments")
        seq = self.evaluate(args[1], local_vars=local_vars)
        return (x for x in seq if self.evaluate((args[0], x), local_vars=local_vars))


    def clopure_remove(self, *args, local_vars):
        if len(args) != 2:
            raise ClopureRuntimeError("remove takes 2 arguments")
        seq = self.evaluate(args[1], local_vars=local_vars)
        return (x for x in seq if not self.evaluate((args[0], x), local_vars=local_vars))


    def clopure_take_while(self, *args, local_vars):
        if len(args) != 2:
            raise ClopureRuntimeError("take-while takes 2 arguments")
        return itertools.takewhile(lambda x: self.evaluate((args[0], x), local_vars=local_vars),
                                                        self.evaluate(args[1], local_vars=local_vars))


    def clopure_pmap(self, *args, local_vars):
        if len(args) <= 1:
            raise ClopureRuntimeError("pmap takes at least 2 arguments")
        seqs = [self.evaluate(arg, local_vars=local_vars) for arg in args[1:]]
        p = Pool(self.threads)
        s = Semaphore(self.queue_size)
        input_iter = (((args[0],) + x, local_vars) for x in input_semaphore_hook(zip(*seqs), s))
        return output_semaphore_hook(p.imap(self.mp_evaluate_wrapper, input_iter), s)


    def clopure_pmap_unord(self, *args, local_vars):
        if len(args) <= 1:
            raise ClopureRuntimeError("pmap-unord takes at least 2 arguments")
        seqs = [self.evaluate(arg, local_vars=local_vars) for arg in args[1:]]
        p = Pool(self.threads)
        s = Semaphore(self.queue_size)
        input_iter = (((args[0],) + x, local_vars) for x in input_semaphore_hook(zip(*seqs), s))
        return output_semaphore_hook(p.imap_unordered(self.mp_evaluate_wrapper, input_iter), s)


    def clopure_iter_mp_split(self, *args, local_vars):
        if len(args) != 1:
            raise ClopureRuntimeError("iter-split takes 1 argument")
        fn = args[0]
        def iter_split_generator(*g):
            q_in = Queue()
            q_out = Queue()
            exit_input_thread = False
            semaphore = Semaphore(self.queue_size)
            ps = [Process(target=self.iter_split_evaluate_wrapper, args=(fn, local_vars, len(g), q_in, q_out)) for i in range(self.threads)]
            for p in ps:
                p.start()
            def input_thread():
                try:
                    for i, item in enumerate(zip(*g)):
                        semaphore.acquire()
                        if exit_input_thread:
                            return
                        q_in.put((i, item))
                except BaseException:
                    traceback.print_exc(file=sys.stdout)
                for i in range(self.threads):
                    q_in.put((0, EOFMessage))

            t = Thread(target=input_thread)
            t.start()
            cur = 0
            n_working_threads = self.threads
            l = [None] * self.queue_size
            while True:
                k, data = q_out.get()
                if data is EOFMessage:
                    n_working_threads -= 1
                    if n_working_threads == 0:
                        break
                    continue
                l[k - cur] = (k, data)
                while l[0]:
                    yield l.pop(0)[1]
                    l.append(None)
                    cur += 1
                    semaphore.release()
            exit_input_thread = True
            semaphore.release()
        return iter_split_generator


    def clopure_iter_mp_split_unord(self, *args, local_vars):
        if len(args) != 1:
            raise ClopureRuntimeError("iter-split-unord takes 1 argument")
        fn = args[0]
        def iter_split_generator(*g):
            q_in = Queue()
            q_out = Queue()
            exit_input_thread = False
            semaphore = Semaphore(self.queue_size)
            ps = [Process(target=self.iter_split_evaluate_wrapper, args=(fn, local_vars, len(g), q_in, q_out)) for i in range(self.threads)]
            for p in ps:
                p.start()
            def input_thread():
                try:
                    for i, item in enumerate(zip(*g)):
                        semaphore.acquire()
                        if exit_input_thread:
                            return
                        q_in.put((i, item))
                except BaseException:
                    traceback.print_exc(file=sys.stdout)
                for i in range(self.threads):
                    q_in.put((0, EOFMessage))
            t = Thread(target=input_thread)
            t.start()
            n_working_threads = self.threads
            while True:
                k, data = q_out.get()
                if data is EOFMessage:
                    n_working_threads -= 1
                    if n_working_threads == 0:
                        break
                    continue
                yield data
                semaphore.release()
            for p in ps:
                p.join()
            exit_input_thread = True
            semaphore.release()
        return iter_split_generator


    def clopure_and(self, *args, local_vars):
        if len(args) == 0:
            raise ClopureRuntimeError("or takes at least 1 argument")
        for arg in args:
            result = self.evaluate(arg, local_vars=local_vars)
            if not result:
                return result
        return result


    def clopure_or(self, *args, local_vars):
        if len(args) == 0:
            raise ClopureRuntimeError("or takes at least 1 argument")
        for arg in args:
            result = self.evaluate(arg, local_vars=local_vars)
            if result:
                return result
        return result


    def clopure_member(self, *args, local_vars):
        if len(args) == 0:
            raise ClopureRuntimeError(". takes at least 1 argument")
        obj = self.evaluate(args[0], local_vars=local_vars)
        for arg in args[1:]:
            if not isinstance(arg, ClopureSymbol):
                raise ClopureRuntimeError("%s is not a symbol" % str(arg))
            obj = getattr(obj, arg.symbol)
        return obj
