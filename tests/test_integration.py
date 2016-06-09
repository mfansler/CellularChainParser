from unittest import TestCase, main

from support_functions import chain_integrate, derivative, expand_tuple_list, list_mod
from Coalgebra import Coalgebra


class ChainIntegrationTestCase(TestCase):

    BR_C = Coalgebra(
        {0: ['v_{1}', 'v_{2}', 'v_{3}', 'v_{4}', 'v_{5}', 'v_{6}', 'v_{7}', 'v_{8}', 'v_{9}', 'v_{10}', 'v_{11}'], 1: ['m_{1}', 'm_{2}', 'm_{3}', 'm_{4}', 'm_{5}', 'm_{6}', 'm_{7}', 'm_{8}', 'm_{9}', 'm_{10}', 'm_{11}', 'm_{12}', 'm_{13}', 'm_{14}', 'c_{1}', 'c_{2}', 'c_{3}', 'c_{4}', 'c_{5}', 'c_{6}', 'c_{7}', 'c_{8}', 'c_{9}', 'c_{10}', 'c_{11}', 'c_{12}', 'c_{13}', 'c_{14}', 'c_{15}', 'c_{16}', 'c_{17}', 'c_{18}'], 2: ['a_{1}', 'a_{2}', 'a_{3}', 'a_{4}', 'e_{1}', 'e_{2}', 's_{1}', 's_{2}', 's_{3}', 's_{4}', 's_{5}', 's_{6}', 's_{7}', 's_{8}', 's_{9}', 's_{10}', 's_{11}', 's_{12}', 't_{1}', 't_{2}', 't_{3}', 't_{4}', 't_{5}', 't_{6}', 't_{7}', 't_{8}'], 3: ['D', 'q_{1}', 'q_{2}', 'q_{3}', 'q_{4}']},
        {'m_{1}': {'v_{11}': 1, 'v_{1}': 1}, 'c_{5}': {'v_{7}': 1, 'v_{8}': 1}, 'm_{3}': {'v_{11}': 1, 'v_{8}': 1}, 'c_{13}': {'v_{11}': 1, 'v_{10}': 1}, 'c_{7}': {'v_{2}': 1, 'v_{3}': 1}, 'c_{3}': {'v_{2}': 1, 'v_{3}': 1}, 't_{6}': {'m_{7}': 1, 'c_{2}': 1, 'm_{6}': 1, 'c_{5}': 1, 'c_{6}': 1}, 't_{2}': {'c_{3}': 1, 'm_{5}': 1, 'c_{4}': 1, 'c_{1}': 1, 'm_{4}': 1}, 's_{4}': {'m_{9}': 1, 'c_{16}': 1, 'm_{12}': 1, 'c_{9}': 1}, 'c_{1}': {'v_{5}': 1, 'v_{1}': 1}, 'm_{9}': {'v_{7}': 1, 'v_{6}': 1}, 't_{4}': {'c_{7}': 1, 'c_{8}': 1, 'm_{5}': 1, 'c_{11}': 1, 'm_{4}': 1}, 's_{6}': {'c_{10}': 1, 'm_{7}': 1, 'c_{14}': 1, 'm_{3}': 1}, 'e_{2}': {'c_{18}': 1, 'c_{16}': 1, 'c_{14}': 1, 'c_{11}': 1, 'c_{12}': 1}, 'a_{3}': {'m_{1}': 1, 'c_{17}': 1}, 't_{7}': {'m_{9}': 1, 'c_{10}': 1, 'm_{8}': 1, 'c_{9}': 1, 'c_{12}': 1}, 'm_{12}': {'v_{5}': 1, 'v_{8}': 1}, 'c_{11}': {'v_{5}': 1, 'v_{1}': 1}, 'c_{15}': {'v_{5}': 1, 'v_{6}': 1}, 'a_{1}': {'c_{17}': 1, 'm_{14}': 1}, 'q_{3}': {'a_{3}': 1, 's_{1}': 1, 'e_{1}': 1, 's_{5}': 1, 't_{2}': 1, 's_{11}': 1, 't_{6}': 1}, 'c_{9}': {'v_{7}': 1, 'v_{8}': 1}, 't_{8}': {'c_{10}': 1, 'm_{7}': 1, 'm_{6}': 1, 'c_{9}': 1, 'c_{12}': 1}, 'c_{17}': {'v_{11}': 1, 'v_{1}': 1}, 's_{8}': {'c_{8}': 1, 'm_{13}': 1, 'm_{10}': 1, 'm_{12}': 1, 'c_{10}': 1}, 'c_{16}': {'v_{5}': 1, 'v_{6}': 1}, 'm_{10}': {'v_{4}': 1, 'v_{5}': 1}, 's_{11}': {'m_{2}': 1, 'c_{15}': 1, 'm_{5}': 1, 'c_{4}': 1}, 'm_{2}': {'v_{3}': 1, 'v_{6}': 1}, 'm_{14}': {'v_{11}': 1, 'v_{1}': 1}, 'm_{6}': {'v_{7}': 1, 'v_{6}': 1}, 'q_{2}': {'e_{2}': 1, 'a_{2}': 1, 's_{10}': 1, 's_{8}': 1, 's_{4}': 1, 't_{3}': 1, 't_{7}': 1}, 's_{3}': {'m_{9}': 1, 'c_{5}': 1, 'c_{15}': 1, 'm_{12}': 1}, 'q_{4}': {'e_{2}': 1, 's_{2}': 1, 't_{8}': 1, 'a_{4}': 1, 's_{6}': 1, 't_{4}': 1, 's_{12}': 1}, 'D': {'a_{3}': 1, 'a_{4}': 1, 'a_{1}': 1, 'a_{2}': 1}, 'c_{10}': {'v_{8}': 1, 'v_{9}': 1}, 't_{1}': {'m_{11}': 1, 'c_{3}': 1, 'm_{10}': 1, 'c_{4}': 1, 'c_{1}': 1}, 'c_{6}': {'v_{8}': 1, 'v_{9}': 1}, 't_{5}': {'m_{9}': 1, 'c_{5}': 1, 'm_{8}': 1, 'c_{2}': 1, 'c_{6}': 1}, 's_{5}': {'m_{7}': 1, 'c_{13}': 1, 'm_{3}': 1, 'c_{6}': 1}, 's_{2}': {'m_{1}': 1, 'm_{3}': 1, 'c_{7}': 1, 'c_{9}': 1, 'm_{6}': 1, 'm_{2}': 1, 'm_{4}': 1}, 't_{3}': {'m_{11}': 1, 'c_{8}': 1, 'm_{10}': 1, 'c_{7}': 1, 'c_{11}': 1}, 'm_{13}': {'v_{3}': 1, 'v_{9}': 1}, 'm_{8}': {'v_{10}': 1, 'v_{9}': 1}, 's_{7}': {'m_{13}': 1, 'm_{10}': 1, 'm_{12}': 1, 'c_{6}': 1, 'c_{4}': 1}, 'm_{11}': {'v_{2}': 1, 'v_{1}': 1}, 'c_{8}': {'v_{4}': 1, 'v_{3}': 1}, 'c_{4}': {'v_{4}': 1, 'v_{3}': 1}, 'a_{2}': {'c_{18}': 1, 'm_{14}': 1}, 's_{1}': {'m_{1}': 1, 'c_{5}': 1, 'm_{3}': 1, 'c_{3}': 1, 'm_{6}': 1, 'm_{2}': 1, 'm_{4}': 1}, 'e_{1}': {'c_{17}': 1, 'c_{15}': 1, 'c_{2}': 1, 'c_{13}': 1, 'c_{1}': 1}, 'q_{1}': {'s_{3}': 1, 'e_{1}': 1, 'a_{1}': 1, 't_{1}': 1, 't_{5}': 1, 's_{7}': 1, 's_{9}': 1}, 'c_{2}': {'v_{10}': 1, 'v_{6}': 1}, 'c_{12}': {'v_{10}': 1, 'v_{6}': 1}, 's_{10}': {'m_{11}': 1, 'c_{7}': 1, 'c_{14}': 1, 'm_{13}': 1, 'm_{8}': 1, 'm_{14}': 1}, 'm_{7}': {'v_{10}': 1, 'v_{9}': 1}, 'a_{4}': {'m_{1}': 1, 'c_{18}': 1}, 'c_{14}': {'v_{11}': 1, 'v_{10}': 1}, 'c_{18}': {'v_{11}': 1, 'v_{1}': 1}, 'm_{4}': {'v_{2}': 1, 'v_{1}': 1}, 'm_{5}': {'v_{4}': 1, 'v_{5}': 1}, 's_{9}': {'m_{11}': 1, 'c_{13}': 1, 'c_{3}': 1, 'm_{13}': 1, 'm_{8}': 1, 'm_{14}': 1}, 's_{12}': {'m_{2}': 1, 'c_{16}': 1, 'm_{5}': 1, 'c_{8}': 1}},
        {}  # don't really care about coproduct definition
    )

    maxDiff = None

    def test_BR3_g3_h1_0(self):
        g3_h1_0 = [(['m_{11}', 'm_{4}'], ['v_{1}', 'v_{2}'], ['m_{4}']),
                   (['m_{11}', 'm_{4}'], ['m_{4}'], ['v_{1}', 'v_{2}'])]

        anti_g3_h1_0 = chain_integrate(g3_h1_0, self.BR_C)

        self.assertIsNotNone(anti_g3_h1_0)
        self.assertEqual(set(anti_g3_h1_0), {('m_{4}', 'm_{4}', 'm_{4}'),
                                             ('m_{11}', 'm_{4}', 'm_{4}')})

    def test_BR3_g3_h1_1(self):
        g3_h1_1a = [(['c_{3}', 'c_{7}'], ['v_{1}', 'v_{3}'], ['m_{4}']),
                    (['c_{3}', 'c_{7}'], ['c_{3}'], ['v_{1}', 'v_{3}']),
                    (['c_{3}', 'c_{7}'], ['m_{4}'], ['v_{1}', 'v_{2}']),
                    (['c_{3}', 'c_{7}'], ['v_{2}', 'v_{3}'], ['c_{3}'])]

        anti_g3_h1_1a = chain_integrate(g3_h1_1a, self.BR_C)

        self.assertIsNotNone(anti_g3_h1_1a)

        d_anti_g3_h1_1a = [dX for dXs in derivative(anti_g3_h1_1a, self.BR_C) for dX in expand_tuple_list(dXs)]
        d_anti_g3_h1_1a = list_mod(d_anti_g3_h1_1a)
        g3_h1_1a_exp = [x for xs in g3_h1_1a for x in expand_tuple_list(xs)]
        g3_h1_1a_exp = list_mod(g3_h1_1a_exp)
        self.assertSetEqual(set(d_anti_g3_h1_1a), set(g3_h1_1a_exp))

    def test_sum_non_boundary_cycles(self):
        g3_h2_0 = [(['m_{11}', 'm_{4}'], ['m_{8}', 'm_{7}'], ['c_{13}', 'c_{14}']),
                   (['m_{7}', 'm_{8}'], ['c_{14}', 'c_{13}'], ['m_{5}', 'm_{10}']),
                   (['m_{4}', 'm_{11}'], ['m_{9}', 'm_{6}'], ['c_{7}', 'c_{3}']),
                   (['m_{6}', 'm_{9}'], ['c_{3}', 'c_{7}'], ['m_{4}', 'm_{11}'])]

        anti_g3_h2_0 = chain_integrate(g3_h2_0, self.BR_C)
        self.assertIsNotNone(anti_g3_h2_0)

if __name__ == '__main__':
    main()
