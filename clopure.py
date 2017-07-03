#!/usr/bin/env python3

import re
import ast
import readline
import core


clopure_token = re.compile(r"\(|\)|\[|\]|\{|\}|\#\{|\#\(|'|\".+(?<!\\)\"|;|[^\s,\(\)\[\]\{\}\#'\";]+")
clopure_sep = re.compile(r"^[\s,]*$")


class ClopureRunner(object):
    def __init__(self):
        self.global_vars = {}

    def evaluate(self, node):
        return core.evaluate(node, global_vars=self.global_vars, local_vars={})


class ClopureParser(object):
    def __init__(self):
        self.stack = [(None, [])]

    def is_empty(self):
        return len(self.stack) == 1

    def parse_line(self, s):
        current_pos = 0
        for m in clopure_token.finditer(s):
            if not clopure_sep.match(s[current_pos:m.start()]):
                raise Exception("Syntax error")
            token = m.group(0)
            if token == ";":
                break
            elif token == "'":
                self.stack.append(("quote", []))
            elif token == "(":
                self.stack.append(("list", []))
            elif token == "#(":
                self.stack.append(("list-fn", []))
            elif token == "[":
                self.stack.append(("vector", []))
            elif token == "{":
                self.stack.append(("dict", []))
            elif token == "#{":
                self.stack.append(("set", []))
            elif token == ")":
                node = self.stack.pop()
                if node[0] == "list":
                    self.stack[-1][1].append(tuple(node[1]))
                elif node[0] == "list-fn":
                    self.stack[-1][1].append((core.ClopureSymbol("fn"),) + tuple(node[1]))
                else:
                    raise Exception("Syntax error")
            elif token == "]":
                node = self.stack.pop()
                if node[0] == "vector":
                    self.stack[-1][1].append(node[1])
                else:
                    raise Exception("Syntax error")
            elif token == "}":
                node = self.stack.pop()
                if node[0] == "dict":
                    if len(node[1]) % 2 == 0:
                        raise Exception("Syntax error")
                    self.stack[-1][1].append({node[1][i]: node[1][i + 1] for i in range(0, len(node[1]), 2)})
                elif node[0] == "set":
                    self.stack[-1][1].append(set(node[1]))
                else:
                    raise Exception("Syntax error")
            else:
                try:
                    lt = ast.literal_eval(token)
                    self.stack[-1][1].append(lt)
                except (ValueError, SyntaxError):
                    self.stack[-1][1].append(core.ClopureSymbol(token))
            if token != "'":
                if self.stack[-1][0] == "quote":
                    t = self.stack.pop()
                    self.stack[-1][1].append((core.ClopureSymbol("quote"), t[1][0]))

            current_pos = m.end()

        ret = self.stack[0][1][:]
        self.stack[0][1].clear()
        return ret


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
