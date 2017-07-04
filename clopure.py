#!/usr/bin/env python3

import sys
import readline
import traceback

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
    else:
        stream = sys.stdin

    while True:
        if not args.FILE and sys.stdin.isatty():
            try:
                if clparser.is_empty():
                    line = input(">>> ").rstrip()
                else:
                    line = input("... ").rstrip()
            except EOFError:
                print()
                break
        else:
            line = stream.readline()
            if not line:
                break
        try:
            trees = clparser.parse_line(line)
        except ClopureSyntaxError as e:
            print("Syntax error: %s" % str(e), file=sys.stdout)
            print(line, file=sys.stdout)
            print(" " * e.pos + "^", file=sys.stdout)
            clparser.clear()
            continue
        try:
            for tree in trees:
                result = runner.evaluate(tree)
                if not args.FILE and sys.stdin.isatty():
                    print(result)
        except ClopureRuntimeError as e:
            print("ClopureRuntimeError: %s" % str(e), file=sys.stdout)
        except Exception:
            traceback.print_exc(file=sys.stdout)
    if not clparser.is_empty():
        print("Syntax error: Form is not closed", file=sys.stdout)


if __name__ == "__main__":
    main()
