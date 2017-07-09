import unittest
import sys

from io import StringIO
from clopure.core import ClopureRunner
from clopure.parser import ClopureParser


class TestUnit(unittest.TestCase):

    def setUp(self):
        self.parser = ClopureParser()
        self.runner = ClopureRunner()

    def test_doseq(self):
        code = "(doseq [x [1 2 3] y [4 5 6]] (print x y))"
        tree = self.parser.parse_line(code)
        io = StringIO()
        sys.stdout = io
        result = self.runner.evaluate(tree[0])
        sys.stdout = sys.__stdout__
        self.assertEqual(io.getvalue(), "1 4\n1 5\n1 6\n2 4\n2 5\n2 6\n3 4\n3 5\n3 6\n")
