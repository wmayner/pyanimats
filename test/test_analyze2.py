from analyze2 import expand_list_by_func
import unittest

class TestAnalyze2(unittest.TestCase):

    def test_expand_list_by_func(self):
        list_things = range(1, 50)
        columns = {
            'a': lambda x: x,
            'b': lambda x: 2*x,
            'c': lambda x: 1/x,
        }

        correct_table = []
        for thing in list_things:
            correct_table.append({
                'a': thing,
                'b': 2*thing,
                'c': 1/thing,
            })

        test_table = expand_list_by_func(list_things, columns)

        # for i in range(len(list_things)):
        #     assertDictEqual(correct_table, test_table)
        self.assertEqual(correct_table, test_table)
