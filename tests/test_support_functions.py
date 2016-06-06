import numpy as np
from random import randrange
from scipy import sparse as sp
from unittest import TestCase, main

from support_functions import expand_tuple_list, ref_mod2, row_reduce_mod2, backsubstitute_mod2, \
    mat_mod2, select_basis, tensor, add_maps_mod_2


class TensorTestCase(TestCase):
    def test_single_tensor(self):

        X = {0: ['v']}

        X_x_X = {0: {('v', 'v')}}

        self.maxDiff = None
        self.assertEqual(tensor(X, X), X_x_X)

    def test_multidim_tensor(self):

        X = {0: ['v'],
             1: ['a', 'b'],
             2: ['aa', 'ab']}

        X_x_X = {0: {('v', 'v')},
                 1: {('v', 'a'), ('v', 'b'), ('a', 'v'), ('b', 'v')},
                 2: {('v', 'aa'), ('v', 'ab'), ('aa', 'v'), ('ab', 'v'),
                     ('a', 'b'), ('b', 'a'), ('a', 'a'), ('b', 'b')},
                 3: {('a', 'aa'), ('a', 'ab'), ('b', 'aa'), ('b', 'ab'),
                     ('aa', 'a'), ('aa', 'b'), ('ab', 'a'), ('ab', 'b')},
                 4: {('aa', 'aa'), ('aa', 'ab'), ('ab', 'aa'), ('ab', 'ab')}}

        self.maxDiff = None
        self.assertEqual(tensor(X, X), X_x_X)


class LinearSystemSolvingTestCase(TestCase):
    def test_row_echelon_form(self):

        # create matrix
        test_mat = sp.lil_matrix((3, 5), dtype=np.int8)
        test_mat.rows = [[0, 1, 4], [1, 3, 4], [0, 2, 3]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        # row echelon form
        test_mat_ref, rank = ref_mod2(test_mat.copy(), augment=1, eliminate=False)

        # verify rank
        self.assertEqual(rank, 3)

        sol = backsubstitute_mod2(test_mat_ref)

        self.assertEqual(sol, [1])

        colsum = sp.lil_matrix((3, 1), dtype=np.int8)

        for n in sol:
            colsum += test_mat[:, n]

        self.assertFalse(any(colsum != test_mat.getcol(-1)))

    def test_row_echelon_form_all(self):

        # create matrix
        test_mat = sp.lil_matrix((4, 5), dtype=np.int8)
        test_mat.rows = [[2, 4], [0, 4], [3, 4], [1, 4]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        # row echelon form
        test_mat_ref, rank = ref_mod2(test_mat.copy(), augment=1, eliminate=False)

        # verify rank
        self.assertEqual(rank, 4)

        sol = backsubstitute_mod2(test_mat_ref)

        self.assertEqual(sol, [3, 2, 1, 0])

        colsum = sp.lil_matrix((4, 1), dtype=np.int8)

        for n in sol:
            colsum += test_mat[:, n]

        self.assertFalse(any(colsum != test_mat.getcol(-1)))

    def test_larger_row_echelon_form(self):

        # create matrix
        test_mat = sp.lil_matrix((6, 10), dtype=np.int8)
        test_mat.rows = [[6, 9], [5, 6, 9], [0, 2, 6, 7], [2, 3, 6, 8], [0, 1, 5], [4, 5, 6, 7, 9]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        # row echelon form
        test_mat_ref, rank = ref_mod2(test_mat.copy(), augment=1, eliminate=False)

        # verify rank
        self.assertEqual(rank, 6)

        sol = backsubstitute_mod2(test_mat_ref)

        self.assertEqual(sol, [6, 2])

        colsum = sp.lil_matrix((6, 1), dtype=np.int8)

        for n in sol:
            colsum += test_mat[:, n]

        colsum.data = [[x % 2 for x in row] for row in colsum.data]

        self.assertFalse(any(colsum != test_mat.getcol(-1)))

    def test_larger_reduced_row_echelon_form(self):

        # create matrix
        test_mat = sp.lil_matrix((6, 10), dtype=np.int8)
        test_mat.rows = [[6, 9], [5, 6, 9], [0, 2, 6, 7], [2, 3, 6, 8], [0, 1, 5], [4, 5, 6, 7, 9]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        # row echelon form
        test_mat_rref, rank = row_reduce_mod2(test_mat.copy(), augment=1)

        # verify rank
        self.assertEqual(rank, 6)

        sol = backsubstitute_mod2(test_mat_rref)

        self.assertEqual(sol, [6, 2])

        colsum = sp.lil_matrix((6, 1), dtype=np.int8)

        for n in sol:
            colsum += test_mat[:, n]

        colsum.data = [[x % 2 for x in row] for row in colsum.data]

        self.assertFalse(any(colsum != test_mat.getcol(-1)))

    def test_independent_set_small(self):

        # create matrix
        test_mat = sp.lil_matrix((3, 5), dtype=np.int8)
        test_mat.rows = [[0, 1, 4], [1, 3, 4], [4]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        # row echelon form
        test_mat_ref, rank = ref_mod2(test_mat.copy(), augment=1, eliminate=False)

        # verify rank
        self.assertEqual(rank, 2)

        sol = backsubstitute_mod2(test_mat_ref)

        self.assertEqual(sol, [4])

    def test_independent_set_larger(self):

        # create matrix
        test_mat = sp.lil_matrix((6, 10), dtype=np.int8)
        test_mat.rows = [[0, 1, 8], [1, 5], [0, 2, 4], [9], [0, 6], [0, 2, 6, 8]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        # row echelon form
        test_mat_ref, rank = ref_mod2(test_mat.copy(), augment=1, eliminate=True)

        # verify rank
        self.assertEqual(rank, 5)

        sol = backsubstitute_mod2(test_mat_ref)

        self.assertEqual(sol, [9])


class BasisSelectionTestCase(TestCase):
    def test_eye(self):

        test_mat = sp.eye(3)
        cols, rr_test_mat, rank = select_basis(test_mat)

        self.assertEqual(cols, [0, 1, 2])
        self.assertEqual(rank, 3)

    def test_augmented_eye(self):

        test_mat = sp.eye(3)
        test_mat = sp.hstack([test_mat, test_mat])

        cols, rr_test_mat, rank = select_basis(test_mat)

        self.assertEqual(cols, [0, 1, 2])
        self.assertEqual(rank, 3)

    def test_augmented_eye_shuffled(self):

        test_mat = sp.lil_matrix((3, 6), dtype=np.int8)
        test_mat.rows = [[0, 2], [4, 5], [1, 3]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        cols, rr_test_mat, rank = select_basis(test_mat)

        self.assertEqual(cols, [0, 1, 4])
        self.assertEqual(rank, 3)

    def test_blank_first_row(self):
        test_mat = sp.lil_matrix((4, 6), dtype=np.int8)
        test_mat.rows = [[], [0, 2], [4, 5], [1, 3]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        cols, rr_test_mat, rank = select_basis(test_mat)

        self.assertEqual(cols, [0, 1, 4])
        self.assertEqual(rank, 3)

    def test_blank_last_row(self):

        test_mat = sp.lil_matrix((4, 6), dtype=np.int8)
        test_mat.rows = [[0, 2], [4, 5], [1, 3], []]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        cols, rr_test_mat, rank = select_basis(test_mat)

        self.assertEqual(cols, [0, 1, 4])
        self.assertEqual(rank, 3)

    def test_blank_middle_rows(self):

        test_mat = sp.lil_matrix((6, 6), dtype=np.int8)
        test_mat.rows = [[0, 2], [], [4, 5], [], [], [1, 3]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        cols, rr_test_mat, rank = select_basis(test_mat)

        self.assertEqual(cols, [0, 1, 4])
        self.assertEqual(rank, 3)

    def test_lower_triangular(self):

        test_mat = sp.lil_matrix((4, 4), dtype=np.int8)
        test_mat.rows = [[0], [0, 1], [2], [1, 2, 3]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        cols, rr_test_mat, rank = select_basis(test_mat)

        self.assertEqual(cols, [0, 1, 2, 3])
        self.assertEqual(rank, 4)

    def test_upper_triangular(self):

        test_mat = sp.lil_matrix((4, 4), dtype=np.int8)
        test_mat.rows = [[0, 3], [1], [2, 3], [3]]
        test_mat.data = [[1]*len(row) for row in test_mat.rows]

        cols, rr_test_mat, rank = select_basis(test_mat)

        self.assertEqual(cols, [0, 1, 2, 3])
        self.assertEqual(rank, 4)


class ExpandTupleListTestCase(TestCase):
    def test_simple_tuple(self):
        self.assertEqual([(1, 1)], expand_tuple_list(([1], [1])))

    def test_nontuples(self):
        self.assertRaises(TypeError, expand_tuple_list, 1)
        self.assertRaises(TypeError, expand_tuple_list, [1, 2])
        self.assertRaises(TypeError, expand_tuple_list, [([1, 2], [1])])

    def test_two_tuple(self):
        self.assertEqual(expand_tuple_list(([1], [1, 2])), [(1, 1), (1, 2)])
        self.assertEqual(expand_tuple_list(([1, 2], [1])), [(1, 1), (2, 1)])
        self.assertEqual(expand_tuple_list(([1, 2], [1, 2])), [(1, 1), (1, 2), (2, 1), (2, 2)])

    def test_three_tuple(self):
        tp_list = ([1, 2], [1, 2, 3], [5, 6])
        expected = [(1, 1, 5), (1, 1, 6), (1, 2, 5), (1, 2, 6), (1, 3, 5), (1, 3, 6),
                    (2, 1, 5), (2, 1, 6), (2, 2, 5), (2, 2, 6), (2, 3, 5), (2, 3, 6)]
        self.assertEqual(expand_tuple_list(tp_list), expected)

    def test_counts(self):
        self.assertEqual(len(expand_tuple_list((range(3), range(3)))), 9)
        self.assertEqual(len(expand_tuple_list((range(40), range(5)))), 200)
        self.assertEqual(len(expand_tuple_list((range(10), range(11), range(12)))), 1320)
        self.assertEqual(len(expand_tuple_list((range(1), range(2), range(3), range(4), range(5)))), 120)


class AddMapsTestCase(TestCase):
    def test_identical_simple_map(self):
        m1 = {0: [0, 1], 1: [0, 3], 2: [5]}

        self.assertFalse(any(add_maps_mod_2(m1, m1).itervalues()))

    def test_originals_intact(self):
        m1 = {0: [0, 1], 1: [0, 3], 2: [5]}
        m2 = {0: [0, 1], 1: [0, 3], 2: [5]}

        m3 = add_maps_mod_2(m1, m2)

        self.assertDictEqual(m1, {0: [0, 1], 1: [0, 3], 2: [5]})
        self.assertDictEqual(m2, {0: [0, 1], 1: [0, 3], 2: [5]})

    def test_multiple_in_first(self):
        m1 = {0: [3, 3, 3], 1: [2, 2], 2: [4, 4, 4, 4]}
        m2 = {0: [3], 1: [2], 2: [4]}

        m3 = add_maps_mod_2(m1, m2)

        self.assertDictEqual(m3, {0: [], 1: [2], 2: [4]})

    def test_multiple_in_second(self):
        m1 = {0: [3, 3, 3], 1: [2, 2], 2: [4, 4, 4, 4]}
        m2 = {0: [3], 1: [2], 2: [4]}

        m3 = add_maps_mod_2(m2, m1)

        self.assertDictEqual(m3, {0: [], 1: [2], 2: [4]})

    def test_commutative(self):
        m1 = {0: [randrange(5) for i in range(20)]}
        m2 = {0: [randrange(5) for i in range(20)]}

        m3 = add_maps_mod_2(m1, m2)
        m4 = add_maps_mod_2(m2, m1)

        self.assertSetEqual(set(m3[0]), set(m4[0]))


if __name__ == '__main__':
    main()
