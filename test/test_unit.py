import unittest
import sys
import types

from io import StringIO
from clopure.core import ClopureRunner, ClopureSymbol, ClopureFunction
from clopure.parser import ClopureParser


class TestUnit(unittest.TestCase):

    def setUp(self):
        self.parser = ClopureParser()
        self.runner = ClopureRunner()

    def test_import(self):
        code = "(import sys)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, sys)
        code = "((import operator add) 1 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 3)

    def test_defimport(self):
        code = "(defimport sys) sys"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        result = self.runner.evaluate(tree[1])
        self.assertEqual(result, sys)
        code = "(defimport operator add) (add 1 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        result = self.runner.evaluate(tree[1])
        self.assertEqual(result, 3)

    def test_defimport_as(self):
        code = "(defimport-as x sys) x"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        result = self.runner.evaluate(tree[1])
        self.assertEqual(result, sys)
        code = "(defimport-as x operator add) (x 1 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        result = self.runner.evaluate(tree[1])
        self.assertEqual(result, 3)

    def test_def(self):
        code = "(def x [1 2 3]) x"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        result = self.runner.evaluate(tree[1])
        self.assertEqual(result, [1, 2, 3])
        code = "(def x ([1 2 3] 1)) x"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        result = self.runner.evaluate(tree[1])
        self.assertEqual(result, 2)


    def test_let(self):
        code = "(def x 5) (+ (let [x (+ 1 2) y (+ 3 4)] (print (+ x y)) (print (- x y)) (* x y)) x)"
        tree = self.parser.parse_line(code)
        io = StringIO()
        sys.stdout = io
        result = self.runner.evaluate(tree[0])
        result = self.runner.evaluate(tree[1])
        sys.stdout = sys.__stdout__
        self.assertEqual(io.getvalue(), "10\n-4\n")
        self.assertEqual(result, 26)
        code = "(+ (let [x (+ 1 2) y (+ 3 4)] (+ x y) (- x y) (* x y)) y)"
        tree = self.parser.parse_line(code)
        with self.assertRaises(Exception) as cm:
            result = self.runner.evaluate(tree[0])


    def test_fn(self):
        code = "((fn [x y] (+ x y)) 1 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 3)
        code = "(#(+ % %2) 1 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 3)
        code = "(#(+ %1 %2) 1 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 3)
        code = "((fn f [x y] f) 1 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertTrue(isinstance(result, ClopureFunction))

    def test_defn(self):
        code = "(defn f [x y] (+ x y)) (f 1 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        result = self.runner.evaluate(tree[1])
        self.assertEqual(result, 3)

    def test_extract_args(self):
        code = "(extract-args + [1 2 3] [4 5 6])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 21)

    def test_quote(self):
        code = "(quote (1 2 3))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, (1, 2, 3))
        code = "'(1 2 3)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, (1, 2, 3))
        code = "(1 2 3)"
        tree = self.parser.parse_line(code)
        with self.assertRaises(Exception) as cm:
            result = self.runner.evaluate(tree[0])

    def test_eval(self):
        code = "(eval '(+ 1 2))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 3)

    def test_if(self):
        code = "(if 1 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 2)
        code = "(if 0 2)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, None)
        code = "(if 1 2 3)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 2)
        code = "(if 0 2 3)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 3)
        code = "(if (+ 1 2) (+ 3 4) (+ 5 6))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 7)
        code = "(if 0 (+ 3 4) (+ 5 6))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 11)

    def test_do(self):
        code = "(do)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, None)
        code = "(do (print (+ 1 2)) (print (+ 3 4)) (+ 5 6))"
        tree = self.parser.parse_line(code)
        io = StringIO()
        sys.stdout = io
        result = self.runner.evaluate(tree[0])
        sys.stdout = sys.__stdout__
        self.assertEqual(io.getvalue(), "3\n7\n")
        self.assertEqual(result, 11)

    def test_doseq(self):
        code = "(doseq [x [1 2 3] y [4 5 6]] (print x y))"
        tree = self.parser.parse_line(code)
        io = StringIO()
        sys.stdout = io
        result = self.runner.evaluate(tree[0])
        sys.stdout = sys.__stdout__
        self.assertEqual(io.getvalue(), "1 4\n1 5\n1 6\n2 4\n2 5\n2 6\n3 4\n3 5\n3 6\n")

    def test_dorun(self):
        code = "'((print 1 2) (print 3 4) (print 5 6))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, ((ClopureSymbol("print"), 1, 2), (ClopureSymbol("print"), 3, 4), (ClopureSymbol("print"), 5, 6)))
        code = "(dorun '((print 1 2) (print 3 4) (print 5 6)))"
        tree = self.parser.parse_line(code)
        io = StringIO()
        sys.stdout = io
        result = self.runner.evaluate(tree[0])
        sys.stdout = sys.__stdout__
        self.assertEqual(io.getvalue(), "1 2\n3 4\n5 6\n")

    def test_map(self):
        code = "(list (map #(+ %1 %2) [0 1 2 3] [4 5 6 7]))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, [4, 6, 8, 10])
        code = "(map #(+ %1 %2) [0 1 2 3] [4 5 6 7])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertTrue(isinstance(result, types.GeneratorType))

    def test_reduce(self):
        code = "(reduce #(* (+ %1 1) %2) [6 5 4 3])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 435)
        code = "(reduce #(* (+ %1 1) %2) [4 5])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 25)
        code = "(reduce #(* (+ %1 1) %2) 4 [])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 4)
        code = "(reduce #(* (+ %1 1) %2) [5])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 5)
        code = "(reduce #(* (+ %1 1) %2) [])"
        tree = self.parser.parse_line(code)
        with self.assertRaises(Exception) as cm:
            result = self.runner.evaluate(tree[0])
        code = "(reduce + [])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 0)

    def test_reduced(self):
        code = "(reduce #(reduced (- 1)) [6 5 4 3])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, -1)
        code = "(reduce #(reduced (- 1)) 6 [5 4 3])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, -1)
        code = "(reduce #(reduced (- 1)) 6 [5])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, -1)
        code = "(reduce #(reduced (- 1)) 6 [])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 6)
        code = "(reduce #(reduced (- 1)) [5])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 5)

    def test_filter(self):
        code = "(list (filter #(>= % 5) [6 4 8 2 4 7]))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, [6, 8, 7])
        code = "(filter #(>= % 5) [6 4 8 2 4 7])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertTrue(isinstance(result, types.GeneratorType))

    def test_remove(self):
        code = "(list (remove #(>= % 5) [6 4 8 2 4 7]))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, [4, 2, 4])
        code = "(remove #(>= % 5) [6 4 8 2 4 7])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertTrue(isinstance(result, types.GeneratorType))

    def test_take_while(self):
        code = "(list (take-while #(>= % 5) [7 6 5 4 3 2]))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, [7, 6, 5])
        code = "(take-while #(>= % 5) [7 6 5 4 3 2])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertTrue(hasattr(result, "__next__"))

    def test_and(self):
        code = "(and (do (print \"a\") 1) (do (print \"b\") 0) (do (print \"c\") 1))"
        tree = self.parser.parse_line(code)
        io = StringIO()
        sys.stdout = io
        result = self.runner.evaluate(tree[0])
        sys.stdout = sys.__stdout__
        self.assertEqual(result, 0)
        self.assertEqual(io.getvalue(), "a\nb\n")
        code = "(and True False True)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, False)
        code = "(and True 1 4)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 4)

    def test_or(self):
        code = "(or (do (print \"a\") 0) (do (print \"b\") 1) (do (print \"c\") 0))"
        tree = self.parser.parse_line(code)
        io = StringIO()
        sys.stdout = io
        result = self.runner.evaluate(tree[0])
        sys.stdout = sys.__stdout__
        self.assertEqual(result, 1)
        self.assertEqual(io.getvalue(), "a\nb\n")
        code = "(or False True False)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, True)
        code = "(or False None [])"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, [])

    def test_member(self):
        code = "((. \"abc\" encode) \"utf-8\")"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, b"abc")
        code = "(. 2j imag)"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        self.assertEqual(result, 2.0)

    def tearDown(self):
        sys.stdout = sys.__stdout__
