import re
import ast

from clopure.core import ClopureSymbol
from clopure.exceptions import ClopureSyntaxError


clopure_token = re.compile(r"\(|\)|\[|\]|\{|\}|\#\{|\#\(|'|b?\".*?(?<!\\)\"|;|[^\s,\(\)\[\]\{\}\#'\";]+")
clopure_sep = re.compile(r"^[\s,]*$")
clopure_var_percent = re.compile(r"^%([0-9]+)$")


def get_fn_macro_percent(node):
    ret = set()
    if len(node) != 0:
        if node[0] == ClopureSymbol("fn"):
            return ret
        for x in node:
            if isinstance(x, ClopureSymbol):
                m = clopure_var_percent.match(x.symbol)
                if m:
                    ret.add(int(m.group(1)))
            if isinstance(x, tuple):
                ret.update(get_fn_macro_percent(x))
    return ret


class ClopureParser(object):
    def __init__(self):
        self.stack = [(None, [])]

    def is_empty(self):
        return len(self.stack) == 1

    def clear(self):
        self.stack = [(None, [])]

    def parse_line(self, s):
        current_pos = 0
        for m in clopure_token.finditer(s):
            if not clopure_sep.match(s[current_pos:m.start()]):
                raise ClopureSyntaxError("Unexpected characters", pos=m.start())
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
                    args = get_fn_macro_percent(node[1])
                    args.add(0)
                    self.stack[-1][1].append((ClopureSymbol("fn"), [ClopureSymbol("%%%d" % (i + 1)) for i in range(max(args))], tuple(node[1])))
                else:
                    raise ClopureSyntaxError("Corresponding keyword '(' is not found", pos=m.start())
            elif token == "]":
                node = self.stack.pop()
                if node[0] == "vector":
                    self.stack[-1][1].append(node[1])
                else:
                    raise ClopureSyntaxError("Corresponding keyword '[' is not found", pos=m.start())
            elif token == "}":
                node = self.stack.pop()
                if node[0] == "dict":
                    if len(node[1]) % 2 != 0:
                        raise ClopureSyntaxError("Even number of items are expected", pos=current_pos)
                    self.stack[-1][1].append({node[1][i]: node[1][i + 1] for i in range(0, len(node[1]), 2)})
                elif node[0] == "set":
                    self.stack[-1][1].append(set(node[1]))
                else:
                    raise ClopureSyntaxError("Corresponding keyword '{' is not found", pos=m.start())
            else:
                try:
                    lt = ast.literal_eval(token)
                    self.stack[-1][1].append(lt)
                except (ValueError, SyntaxError):
                    if token == "%":
                        token = "%1"
                    self.stack[-1][1].append(ClopureSymbol(token))
            if token != "'":
                if self.stack[-1][0] == "quote":
                    t = self.stack.pop()
                    self.stack[-1][1].append((ClopureSymbol("quote"), t[1][0]))

            current_pos = m.end()

        ret = self.stack[0][1][:]
        self.stack[0][1].clear()
        return ret
