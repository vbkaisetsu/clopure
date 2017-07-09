import unittest
import time

from clopure.core import ClopureRunner
from clopure.parser import ClopureParser


class TestMultiprocessing(unittest.TestCase):

    def setUp(self):
        self.parser = ClopureParser()
        self.runner = ClopureRunner(procs=4)

    def test_pmap(self):
        code = "(defimport time sleep) (list (pmap #(do (sleep %) %) [1.0 0.8 0.5 0.1 0.1 0.3]))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        start_time = time.time()
        result = self.runner.evaluate(tree[1])
        end_time = time.time()
        self.assertEqual(result, [1.0, 0.8, 0.5, 0.1, 0.1, 0.3])
        self.assertTrue(0.95 < end_time - start_time < 1.05)

    def test_pmap_unord(self):
        code = "(defimport time sleep) (list (pmap-unord #(do (sleep %) %) [1.0 0.8 0.5 0.1 0.1 0.3]))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        start_time = time.time()
        result = self.runner.evaluate(tree[1])
        end_time = time.time()
        self.assertEqual(result, [0.1, 0.1, 0.5, 0.3, 0.8, 1.0])
        self.assertTrue(0.95 < end_time - start_time < 1.05)

    def test_iter_mp_split(self):
        code = "(defimport time sleep) (list ((iter-mp-split #(map #(do (sleep %) %) %)) [1.0 0.8 0.5 0.1 0.1 0.3]))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        start_time = time.time()
        result = self.runner.evaluate(tree[1])
        end_time = time.time()
        self.assertEqual(result, [1.0, 0.8, 0.5, 0.1, 0.1, 0.3])
        self.assertTrue(0.95 < end_time - start_time < 1.05)

    def test_iter_mp_split_unord(self):
        code = "(defimport time sleep) (list ((iter-mp-split-unord #(map #(do (sleep %) %) %)) [1.0 0.8 0.5 0.1 0.1 0.3]))"
        tree = self.parser.parse_line(code)
        result = self.runner.evaluate(tree[0])
        start_time = time.time()
        result = self.runner.evaluate(tree[1])
        end_time = time.time()
        self.assertEqual(result, [0.1, 0.1, 0.5, 0.3, 0.8, 1.0])
        self.assertTrue(0.95 < end_time - start_time < 1.05)
