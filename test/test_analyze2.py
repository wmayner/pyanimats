from analyze2 import expand_list_by_func, extract_list_from_dicts
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


    def test_extract_list_from_dicts(self):
        list_dicts = [
            {'a': {'b': [{}, {}, {'c':[{}, {'d':{'e':'v0'}}]}]}},
            {'a': {'b': [{}, {}, {'c':[{}, {'d':{'e':'v1'}}]}]}},
            {'a': {'b': [{}, {}, {'c':[{}, {'d':{'e':'v2'}}]}]}},
            {'a': {'b': [{}, {}, {'c':[{}, {'d':{'e':'v3'}}]}]}},
            {'a': {'b': [{}, {}, {'c':[{}, {'d':{'e':'v4'}}]}]}},
        ]
        correct_list = list(map(lambda x: "v"+str(x), range(5)))
        test_list = extract_list_from_dicts(list_dicts,
                                           'a', 'b', 2, 'c', 1, 'd', 'e')
        self.assertEqual(test_list, correct_list)
        
