#!/usr/bin/env python3

import sys
import readline
import traceback
import os

from argparse import ArgumentParser

from clopure.core import ClopureRunner
from clopure.parser import ClopureParser
from clopure.exceptions import ClopureSyntaxError, ClopureRuntimeError


def main():

    aparser = ArgumentParser()
    aparser.add_argument("FILE", nargs="?", default="", help="Script file")
    aparser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
    aparser.add_argument("-q", "--queues", type=int, default=100, help="Number of queues")
    args = aparser.parse_args()

    clparser = ClopureParser()
    runner = ClopureRunner(threads=args.threads, queue_size=args.queues)

    if args.FILE:
        stream = open(args.FILE, "r")
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.FILE)))
    else:
        stream = sys.stdin
        sys.path.insert(0, os.path.dirname(os.path.abspath(".")))

    linenum = 0
    while True:
        if not args.FILE and sys.stdin.isatty():
            try:
                if clparser.is_empty():
                    print()
                    line = input(">>> ").rstrip()
                else:
                    line = input("... ").rstrip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                clparser.clear()
                print("(KeyboardInterrupt)", file=sys.stdout)
                continue
        else:
            line = stream.readline()
            linenum += 1
            if not line:
                break
        try:
            trees = clparser.parse_line(line)
        except ClopureSyntaxError as e:
            if args.FILE or not sys.stdin.isatty():
                print("At line %d:" % linenum, file=sys.stdout)
            print("Syntax error: %s" % str(e), file=sys.stdout)
            print(line, file=sys.stdout)
            print(" " * e.pos + "^", file=sys.stdout)
            clparser.clear()
            if args.FILE or not sys.stdin.isatty():
                break
            continue
        try:
            for tree in trees:
                result = runner.evaluate(tree)
                if not args.FILE and sys.stdin.isatty() and result is not None:
                    print(result)
        except ClopureRuntimeError as e:
            if args.FILE or not sys.stdin.isatty():
                print("At line %d:" % linenum, file=sys.stdout)
            print("ClopureRuntimeError: %s" % str(e), file=sys.stdout)
            if args.FILE or not sys.stdin.isatty():
                break
        except Exception:
            if args.FILE or not sys.stdin.isatty():
                print("At line %d:" % linenum, file=sys.stdout)
            traceback.print_exc(file=sys.stdout)
            if args.FILE or not sys.stdin.isatty():
                break
        except KeyboardInterrupt:
            if args.FILE or not sys.stdin.isatty():
                print("At line %d:" % linenum, file=sys.stdout)
            print("(KeyboardInterrupt)", file=sys.stdout)
            traceback.print_exc(file=sys.stdout)
            if args.FILE or not sys.stdin.isatty():
                break
    if not clparser.is_empty():
        print("Syntax error: Form is not closed", file=sys.stdout)


if __name__ == "__main__":
    main()
