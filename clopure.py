#!/usr/bin/env python3

import readline

from clopure.core import ClopureRunner
from clopure.parser import ClopureParser


def main():
    clparser = ClopureParser()
    runner = ClopureRunner()

    while True:
        try:
            if clparser.is_empty():
                line = input(">>> ")
            else:
                line = input("... ")
        except EOFError:
            print()
            return
        trees = clparser.parse_line(line)
        for tree in trees:
            print(runner.evaluate(tree))


if __name__ == "__main__":
    main()
