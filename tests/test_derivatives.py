from unittest import TestCase, main

from support_functions import derivative, expand_tuple_list, list_mod
from Coalgebra import Coalgebra


class ToyExampleTestCase(TestCase):

    DGC = Coalgebra(
        {0: ['v'], 1: ['a', 'b'], 2: ['aa', 'ab']},
        {'aa': {'b': 1}},
        {} # don't really care about coproduct definition
    )

    def test_simple_derivative(self):

        cell = 'aa'

        self.assertEqual(derivative(cell, self.DGC), ['b'])

    def test_tuple_derivative(self):

        cell = ('a', 'aa')
        d_cell = derivative(cell, self.DGC)
        d_cell = [tp for c in d_cell for tp in expand_tuple_list(c)]
        self.assertEqual(d_cell, [('a', 'b')])

    def test_tuple_derivative2(self):

        cell = ('aa', 'v', 'aa')
        d_cell = derivative(cell, self.DGC)
        d_cell = [tp for c in d_cell for tp in expand_tuple_list(c)]

        self.assertEqual(d_cell, [('b', 'v', 'aa'), ('aa', 'v', 'b')])

    def test_chain_derivative1(self):

        chain = ['aa', 'ab']

        self.assertEqual(derivative(chain, self.DGC), ['b'])

    def test_chain_derivative2(self):

        chain = [('v', 'aa'), ('v', 'ab'), ('aa', 'v'), ('ab', 'v'),
                 ('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b')]
        d_chain = derivative(chain, self.DGC)
        d_chain = [tp for c in d_chain for tp in expand_tuple_list(c)]

        self.assertEqual(d_chain, [('v', 'b'), ('b', 'v')])


class BR3TestCase(TestCase):

    BR3 = Coalgebra(
        {0: ['v_{1}', 'v_{2}', 'v_{3}', 'v_{4}', 'v_{5}', 'v_{6}', 'v_{7}', 'v_{8}', 'v_{9}', 'v_{10}', 'v_{11}'],
         1: ['m_{1}', 'm_{2}', 'm_{3}', 'm_{4}', 'm_{5}', 'm_{6}', 'm_{7}',
             'm_{8}', 'm_{9}', 'm_{10}', 'm_{11}', 'm_{12}', 'm_{13}', 'm_{14}',
             'c_{1}', 'c_{2}', 'c_{3}', 'c_{4}', 'c_{5}', 'c_{6}', 'c_{7}', 'c_{8}', 'c_{9}',
             'c_{10}', 'c_{11}', 'c_{12}', 'c_{13}', 'c_{14}', 'c_{15}', 'c_{16}', 'c_{17}', 'c_{18}'],
         2: ['a_{1}', 'a_{2}', 'a_{3}', 'a_{4}', 'e_{1}', 'e_{2}',
             's_{1}', 's_{2}', 's_{3}', 's_{4}', 's_{5}', 's_{6}',
             's_{7}', 's_{8}', 's_{9}', 's_{10}', 's_{11}', 's_{12}',
             't_{1}', 't_{2}', 't_{3}', 't_{4}', 't_{5}', 't_{6}', 't_{7}', 't_{8}'],
         3: ['D', 'q_{1}', 'q_{2}', 'q_{3}', 'q_{4}']},
        {'m_{1}': {'v_{11}': 1, 'v_{1}': 1}, 'c_{5}': {'v_{7}': 1, 'v_{8}': 1}, 'm_{3}': {'v_{11}': 1, 'v_{8}': 1},
         'c_{13}': {'v_{11}': 1, 'v_{10}': 1}, 'c_{7}': {'v_{2}': 1, 'v_{3}': 1}, 'c_{3}': {'v_{2}': 1, 'v_{3}': 1},
         't_{6}': {'m_{7}': 1, 'c_{2}': 1, 'm_{6}': 1, 'c_{5}': 1, 'c_{6}': 1},
         't_{2}': {'c_{3}': 1, 'm_{5}': 1, 'c_{4}': 1, 'c_{1}': 1, 'm_{4}': 1},
         's_{4}': {'m_{9}': 1, 'c_{16}': 1, 'm_{12}': 1, 'c_{9}': 1}, 'c_{1}': {'v_{5}': 1, 'v_{1}': 1},
         'm_{9}': {'v_{7}': 1, 'v_{6}': 1}, 't_{4}': {'c_{7}': 1, 'c_{8}': 1, 'm_{5}': 1, 'c_{11}': 1, 'm_{4}': 1},
         's_{6}': {'c_{10}': 1, 'm_{7}': 1, 'c_{14}': 1, 'm_{3}': 1},
         'e_{2}': {'c_{18}': 1, 'c_{16}': 1, 'c_{14}': 1, 'c_{11}': 1, 'c_{12}': 1}, 'a_{3}': {'m_{1}': 1, 'c_{17}': 1},
         't_{7}': {'m_{9}': 1, 'c_{10}': 1, 'm_{8}': 1, 'c_{9}': 1, 'c_{12}': 1}, 'm_{12}': {'v_{5}': 1, 'v_{8}': 1},
         'c_{11}': {'v_{5}': 1, 'v_{1}': 1}, 'c_{15}': {'v_{5}': 1, 'v_{6}': 1}, 'a_{1}': {'c_{17}': 1, 'm_{14}': 1},
         'q_{3}': {'a_{3}': 1, 's_{1}': 1, 'e_{1}': 1, 's_{5}': 1, 't_{2}': 1, 's_{11}': 1, 't_{6}': 1},
         'c_{9}': {'v_{7}': 1, 'v_{8}': 1}, 't_{8}': {'c_{10}': 1, 'm_{7}': 1, 'm_{6}': 1, 'c_{9}': 1, 'c_{12}': 1},
         'c_{17}': {'v_{11}': 1, 'v_{1}': 1}, 's_{8}': {'c_{8}': 1, 'm_{13}': 1, 'm_{10}': 1, 'm_{12}': 1, 'c_{10}': 1},
         'c_{16}': {'v_{5}': 1, 'v_{6}': 1}, 'm_{10}': {'v_{4}': 1, 'v_{5}': 1},
         's_{11}': {'m_{2}': 1, 'c_{15}': 1, 'm_{5}': 1, 'c_{4}': 1}, 'm_{2}': {'v_{3}': 1, 'v_{6}': 1},
         'm_{14}': {'v_{11}': 1, 'v_{1}': 1}, 'm_{6}': {'v_{7}': 1, 'v_{6}': 1},
         'q_{2}': {'e_{2}': 1, 'a_{2}': 1, 's_{10}': 1, 's_{8}': 1, 's_{4}': 1, 't_{3}': 1, 't_{7}': 1},
         's_{3}': {'m_{9}': 1, 'c_{5}': 1, 'c_{15}': 1, 'm_{12}': 1},
         'q_{4}': {'e_{2}': 1, 's_{2}': 1, 't_{8}': 1, 'a_{4}': 1, 's_{6}': 1, 't_{4}': 1, 's_{12}': 1},
         'D': {'a_{3}': 1, 'a_{4}': 1, 'a_{1}': 1, 'a_{2}': 1}, 'c_{10}': {'v_{8}': 1, 'v_{9}': 1},
         't_{1}': {'m_{11}': 1, 'c_{3}': 1, 'm_{10}': 1, 'c_{4}': 1, 'c_{1}': 1}, 'c_{6}': {'v_{8}': 1, 'v_{9}': 1},
         't_{5}': {'m_{9}': 1, 'c_{5}': 1, 'm_{8}': 1, 'c_{2}': 1, 'c_{6}': 1},
         's_{5}': {'m_{7}': 1, 'c_{13}': 1, 'm_{3}': 1, 'c_{6}': 1},
         's_{2}': {'m_{1}': 1, 'm_{3}': 1, 'c_{7}': 1, 'c_{9}': 1, 'm_{6}': 1, 'm_{2}': 1, 'm_{4}': 1},
         't_{3}': {'m_{11}': 1, 'c_{8}': 1, 'm_{10}': 1, 'c_{7}': 1, 'c_{11}': 1}, 'm_{13}': {'v_{3}': 1, 'v_{9}': 1},
         'm_{8}': {'v_{10}': 1, 'v_{9}': 1}, 's_{7}': {'m_{13}': 1, 'm_{10}': 1, 'm_{12}': 1, 'c_{6}': 1, 'c_{4}': 1},
         'm_{11}': {'v_{2}': 1, 'v_{1}': 1}, 'c_{8}': {'v_{4}': 1, 'v_{3}': 1}, 'c_{4}': {'v_{4}': 1, 'v_{3}': 1},
         'a_{2}': {'c_{18}': 1, 'm_{14}': 1},
         's_{1}': {'m_{1}': 1, 'c_{5}': 1, 'm_{3}': 1, 'c_{3}': 1, 'm_{6}': 1, 'm_{2}': 1, 'm_{4}': 1},
         'e_{1}': {'c_{17}': 1, 'c_{15}': 1, 'c_{2}': 1, 'c_{13}': 1, 'c_{1}': 1},
         'q_{1}': {'s_{3}': 1, 'e_{1}': 1, 'a_{1}': 1, 't_{1}': 1, 't_{5}': 1, 's_{7}': 1, 's_{9}': 1},
         'c_{2}': {'v_{10}': 1, 'v_{6}': 1}, 'c_{12}': {'v_{10}': 1, 'v_{6}': 1},
         's_{10}': {'m_{11}': 1, 'c_{7}': 1, 'c_{14}': 1, 'm_{13}': 1, 'm_{8}': 1, 'm_{14}': 1},
         'm_{7}': {'v_{10}': 1, 'v_{9}': 1}, 'a_{4}': {'m_{1}': 1, 'c_{18}': 1}, 'c_{14}': {'v_{11}': 1, 'v_{10}': 1},
         'c_{18}': {'v_{11}': 1, 'v_{1}': 1}, 'm_{4}': {'v_{2}': 1, 'v_{1}': 1}, 'm_{5}': {'v_{4}': 1, 'v_{5}': 1},
         's_{9}': {'m_{11}': 1, 'c_{13}': 1, 'c_{3}': 1, 'm_{13}': 1, 'm_{8}': 1, 'm_{14}': 1},
         's_{12}': {'m_{2}': 1, 'c_{16}': 1, 'm_{5}': 1, 'c_{8}': 1}
         },
        {}  # don't really care about coproduct definition
    )

    def test_simple_derivative(self):

        cell1 = 'm_{4}'

        self.assertSetEqual(set(derivative(cell1, self.BR3)), set(['v_{1}', 'v_{2}']))

    def test_tuple_derivative(self):

        cell = ('m_{4}', 'm_{11}')
        d_cell = derivative(cell, self.BR3)

        self.assertEqual(d_cell, [(['v_{2}', 'v_{1}'], 'm_{11}'),
                                  ('m_{4}', ['v_{2}', 'v_{1}'])])

        d_cell = [tp for c in d_cell for tp in expand_tuple_list(c)]
        self.assertEqual(d_cell, [('v_{2}', 'm_{11}'), ('v_{1}', 'm_{11}'),
                                  ('m_{4}', 'v_{2}'), ('m_{4}', 'v_{1}')])

    def test_tuple_derivative2(self):

        cell = ('m_{4}', 'm_{11}', 'v_{10}')
        d_cell = derivative(cell, self.BR3)

        self.assertEqual(d_cell, [(['v_{2}', 'v_{1}'], 'm_{11}', 'v_{10}'),
                                  ('m_{4}', ['v_{2}', 'v_{1}'], 'v_{10}'),
                                  ('m_{4}', 'm_{11}', [])])

        d_cell = [tp for c in d_cell for tp in expand_tuple_list(c)]

        self.assertEqual(d_cell, [('v_{2}', 'm_{11}', 'v_{10}'),
                                  ('v_{1}', 'm_{11}', 'v_{10}'),
                                  ('m_{4}', 'v_{2}', 'v_{10}'),
                                  ('m_{4}', 'v_{1}', 'v_{10}')])

    def test_chain_derivative1(self):

        chain = ['m_{4}', 'm_{11}']

        d_chain = derivative(chain, self.BR3)
        self.assertEqual(d_chain, ['v_{2}', 'v_{1}', 'v_{2}', 'v_{1}'])

        d_chain = list_mod(d_chain)
        self.assertFalse(d_chain)

    def test_chain_derivative2(self):

        chain = [('v_{1}', 'm_{4}'), ('v_{2}', 'm_{4}'),
                 ('v_{1}', 'm_{11}'), ('v_{2}', 'm_{11}')]

        d_chain = derivative(chain, self.BR3)
        self.assertEqual(d_chain, [([], 'm_{4}'), ('v_{1}', ['v_{2}', 'v_{1}']), ([], 'm_{4}'), ('v_{2}', ['v_{2}', 'v_{1}']),
                                   ([], 'm_{11}'), ('v_{1}', ['v_{2}', 'v_{1}']), ([], 'm_{11}'), ('v_{2}', ['v_{2}', 'v_{1}'])])

        d_chain = [tp for c in d_chain for tp in expand_tuple_list(c)]

        self.assertEqual(d_chain, [('v_{1}', 'v_{2}'), ('v_{1}', 'v_{1}'), ('v_{2}', 'v_{2}'), ('v_{2}', 'v_{1}'),
                                   ('v_{1}', 'v_{2}'), ('v_{1}', 'v_{1}'), ('v_{2}', 'v_{2}'), ('v_{2}', 'v_{1}')])

        d_chain = list_mod(d_chain)
        self.assertFalse(d_chain)

if __name__ == '__main__':
    main()
