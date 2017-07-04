#!/usr/bin/env python3

import sys
import readline
import traceback

from clopure.core import ClopureRunner
from clopure.parser import ClopureParser
from clopure.exceptions import ClopureSyntaxError, ClopureRuntimeError


def main():
    clparser = ClopureParser()
    runner = ClopureRunner()

    while True:
        try:
            if clparser.is_empty():
                line = input(">>> ").rstrip()
            else:
                line = input("... ").rstrip()
        except EOFError:
            print()
            return
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
                print(result)
        except ClopureRuntimeError as e:
            print("ClopureRuntimeError: %s" % str(e), file=sys.stdout)
        except Exception:
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
