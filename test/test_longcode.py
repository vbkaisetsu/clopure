import unittest

from clopure.core import ClopureRunner
from clopure.parser import ClopureParser


class TestLongCodes(unittest.TestCase):

    def setUp(self):
        self.parser = ClopureParser()
        self.runner = ClopureRunner()

    def test_func_loop(self):
        code = "((fn fact [n] (if (> n 0) (* n (fact (- n 1))) 1)) 5)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 120)
