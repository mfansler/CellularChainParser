import numpy
import scipy.sparse as sp

from collections import Counter
from itertools import combinations
from copy import deepcopy

from Coalgebra import Coalgebra
from factorize import factorize, expand_tuple_list, expand_map_all

__author__ = 'mfansler'


def add_maps_mod_2(a, b):

    res = deepcopy(a)
    for k, vals in deepcopy(b).items():
        if k not in res:
            res[k] = vals
        else:
            for v in vals:
                if v in res[k]:
                    res[k] = [u for u in res[k] if u != v]
                else:
                    res[k].append(v)
    return res


def row_swap(A, r1, r2):
    A.rows[r1], A.rows[r2] = A.rows[r2], A.rows[r1]
    A.data[r1], A.data[r2] = A.data[r2], A.data[r1]


def mat_mod2(A):

    A.data[:] = numpy.fmod(A.data, 2)
    return A


def add_rows(A, r1, r2):
    A.rows[r2] = sorted(list_mod(A.rows[r1] + A.rows[r2]))
    A.data[r2] = [1]*len(A.rows[r2])


def row_reduce_mod2(A, augment=0):

    if A.ndim != 2:
        raise Exception("require two dimensional matrix input, found ", A.ndim)

    A = A.tolil()
    A.data = [[x % 2 for x in xs] for xs in A.data]
    rank = 0
    for i in range(A.shape[1] - augment):

        nzs = A.getcol(i).nonzero()[0]
        upper_nzs = [nz for nz in nzs if nz < rank]
        lower_nzs = [nz for nz in nzs if nz >= rank]

        if len(lower_nzs) > 0:

            row_swap(A, rank, lower_nzs[0])

            for nz in lower_nzs[1:]:
                add_rows(A, rank, nz)

            if rank > 0:
                for nz in upper_nzs:
                    add_rows(A, rank, nz)

            rank += 1

    return A, rank


def list_mod(ls, modulus=2):
    return [s for s, num in Counter(ls).items() if num % modulus]


def chain_map_mod(xs, modulus=2):
    return {k: list_mod(ls, modulus=2) for k, ls in xs.items()}


def facet_to_cells(facet, C):
    return [face for face, facets in C.differential.items() if facet in facets and facets[facet] % 2]


def derivative(x, C):
    if type(x) is list:
        return [dy for y in x for dy in derivative(y, C)]
    if type(x) is tuple:
        return [tuple(x[:i]) + (derivative(x[i], C), ) + tuple(x[i + 1:]) for i in range(len(x))]
    if type(x) is dict:
        dx = {k: [] for k in x.keys()}

        # first handle raising cell-selecting map
        for k, vs in x.items():
            for cell in facet_to_cells(k, C):
                if cell in dx:
                    dx[cell] += vs
                else:
                    dx[cell] = vs

            dx[k] += derivative(vs, C)
        return dx
    if x in C.differential:
        return [k for k, v in C.differential[x].items() if v % 2]

    return []


# generates all 0-n combinations of elements in the list xs
def all_combinations(xs):
    for i in range(len(xs) + 1):
        for c in combinations(xs, i):
            yield c


# generates the function f: C -> H
# @param C Coalgebra to map from
# @param g map from H (== H*(C)) to class representatives in C
#
# returns function f(x)
def generate_f_integral(C, g):

    # create a map from cells to index
    basis = {el: i for (i, el) in enumerate([el for grp in C.groups.values() for el in grp])}
    inv_basis = {v: k for k, v in basis.items()}

    # store num cells
    n = len(basis)

    # prepare n x n incidence matrix
    # includes multiple dimensions, but it's sparse, so no efficiency lost
    inc_mat = sp.lil_matrix((n, n), dtype=numpy.int8)

    # enter incidences
    for el, bd in C.differential.items():
        inc_mat.rows[basis[el]] = [basis[c] for c, i in bd.items() if i % 2]
    inc_mat.data = [[1]*len(row) for row in inc_mat.rows]

    # switch to cols
    inc_mat = inc_mat.transpose()

    # append identity
    inc_mat = sp.hstack([inc_mat, sp.identity(n, dtype=numpy.int8)])

    # row reduce
    rref_mat, rank = row_reduce_mod2(inc_mat, augment=n)

    # extract just the (partial) inverse
    inv_mat = rref_mat.tocsc()[:, n:].tocsr()

    ker_basis = []
    for i in range(rank):
        first = rref_mat[i, :].nonzero()[1][0]
        ker_basis.append(inv_basis[first])

    # clean up
    del inc_mat, rref_mat

    # method to check if chain (typically a cycle) is in Im[boundary]
    # zombies are components that don't get killed by boundary
    def has_zombies(x):

        # convert to vector in established basis
        x_vec = [0]*n
        for el in x:
            x_vec[basis[el]] = 1

        # converts vector to Im[boundary] basis
        zombies = inv_mat.dot(numpy.array(x_vec))[rank:]

        # linear combination of bounding cells
        #print "boundary map: ", [b for (b, v) in zip(ker_basis, inv_mat.dot(numpy.array(x_vec))[:rank]) if v % 2]

        # return true if there are components not spanned by Im[boundary]
        return numpy.fmod(zombies, 2).any()

    # method to be returned
    # cannonical map of x in C to coset in H
    def f(x):

        if type(x) is not list:
            x = [x]

        # check if not cycle (non-vanishing boundary)
        bd = list_mod([dx for cell in x if cell in C.differential for dx in C.differential[cell].items()], 2)
        if bd:
            return []

        # check if killed by known boundaries
        if not has_zombies(x):
            return []

        # TODO: check to see if single elements are sufficient
        # determine what combination of known cycles it corresponds to
        for ks in all_combinations(g.keys()):
            gens = [gen_comp for k in ks for gen_comp in g[k]]

            if not has_zombies(list_mod(gens + x, 2)):
                return list(ks)

        raise Exception("Error: could not find coset!\n", x)

    def integrate1(x):

        if type(x) is not list:
            x = [x]
        # convert to vector in established basis
        x_vec = [0]*n
        for el in x:
            x_vec[basis[el]] = 1

        # converts vector to Im[boundary] basis
        x_ker = inv_mat.dot(numpy.array(x_vec)) % 2

        if any(x_ker[rank:]):
            return None
            # print "WARNING: Invalid integral!"
            # print x, "contains non-vanishing component (cycle)"

        # returns boundary cells that contain x as boundary
        return [b for (b, v) in zip(ker_basis, x_ker[:rank]) if v % 2]

    def integrate(xs):

        # if a single tuple is passed, treat it as a list
        if type(xs) is tuple:
            xs = [xs]

        # if it is a dict, assume it's a chain map
        if type(xs) is dict:
            return integrate_chain_map(xs)

        # if not a list, assume it's an individual element, so push through integrate1
        if type(xs) is not list:
            return integrate1([xs])

        # if it is a list, but is empty, then return empty list
        if not len(xs):
            return []

        # it is a list, but not of tuples
        # so assume that it is list of elements
        if type(xs[0]) is not tuple:
            return integrate1(xs)

        # if it is a list of tuples, but the tuples have lists
        # we'll need to expand out everything
        if type(xs[0][0]) is list:
            expanded_xs = [tp for x in xs for tp in expand_tuple_list(x)]

        # otherwise, we need to factorize it, which improves performance
        else:
            expanded_xs = xs
            xs = factorize(xs)

        # we now have a none empty list of tuples with list components
        # figure out which component in the first tuple can be integrated
        for i, x_cmp in enumerate(xs[0]):

            anti_x_cmp = integrate1(x_cmp)

            # if this component can't be integrated, continue the loop
            if anti_x_cmp is None:
                continue

            # otherwise construct the anti_derivative that kills it
            else:
                if i == 0:
                    anti_x = (anti_x_cmp,) + tuple(xs[0][1:])
                else:
                    anti_x = tuple(xs[0][:i]) + (anti_x_cmp, ) + tuple(xs[0][i + 1:])

                anti_x = expand_tuple_list(anti_x)
                # take the derivative of that anti-derivative and subtract from our list
                anti_x_full_derivative = [dx_tp for dx in derivative(anti_x, C) for dx_tp in expand_tuple_list(dx) if all(dx)]
                remainder = list_mod(anti_x_full_derivative + expanded_xs, 2)

                # attempt to integrate that remaining portion on its own
                anti_rem = integrate(remainder)

                # if successful, then we have constructed a valid integral for xs
                if anti_rem is not None:
                    # sweet
                    return anti_rem + anti_x

                # otherwise loop back around and check for another component

        return None

    def integrate_chain_map(xs, allow_regress=False):
        # assuming map comes in factored
        expanded_map = chain_map_mod(expand_map_all(xs))
        best_distance = sum([len(vs) for vs in expanded_map.values()])
        #print "starting distance = ", best_distance
        if best_distance == 0:
            return {}

        frontier = []

        # generate initial frontier
        for cell, tps in expanded_map.items():
            for tp in tps:

                # first generate anti-derivatives that kill chain components
                for i in range(len(tp)):
                    for anti_tp_cmp in facet_to_cells(tp[i], C):
                        anti_tp_map = {cell: [tp[:i] + (anti_tp_cmp,) + tp[i+1:]]}
                        der_anti_tp_map = chain_map_mod(expand_map_all(derivative(anti_tp_map, C)))
                        der_size = sum([len(vs) for vs in der_anti_tp_map.values()])
                        num_hits = sum([1 for k, vs in der_anti_tp_map.items() for v in vs if k in expanded_map and v in expanded_map[k]])
                        distance = best_distance + der_size - 2*num_hits
                        if distance == 0:
                            return anti_tp_map
                        if allow_regress or distance < best_distance:
                            frontier.append((distance, anti_tp_map, der_anti_tp_map))

                # next, generate boundary components
                for d_cell in derivative(cell, C):
                    anti_tp_map = {d_cell: [tp]}
                    der_anti_tp_map = chain_map_mod(expand_map_all(derivative(anti_tp_map, C)))
                    der_size = sum([len(vs) for vs in der_anti_tp_map.values()])
                    num_hits = sum([1 for k, vs in der_anti_tp_map.items() for v in vs if k in expanded_map and v in expanded_map[k]])
                    distance = best_distance + der_size - 2*num_hits
                    if distance == 0:
                        return anti_tp_map
                    if allow_regress or distance < best_distance:
                        frontier.append((distance, anti_tp_map, der_anti_tp_map))

        frontier = sorted(frontier, key=lambda tp: tp[0])
        result = None
        while result is None and frontier:
            cur_anti_xs = frontier.pop(0)
            result = integrate_chain_map(add_maps_mod_2(expanded_map, cur_anti_xs[2]), allow_regress=False)

        if result is None:
            return None

        return add_maps_mod_2(cur_anti_xs[1], result)

    return f, integrate


def main():

    print "Expand Tuple List tests"
    print "([1], [1, 2]) = ", expand_tuple_list(([1], [1, 2]))
    print "([1, 2], [1, 2, 3], [5, 6]) =", expand_tuple_list(([1, 2], [1, 2, 3], [5, 6]))
    print

    # test data toy
    DGC = Coalgebra(
        {0: ['v'], 1: ['a', 'b'], 2: ['aa', 'ab']},
        {'aa': {'b': 1}},
        {} # don't really care about coproduct definition

    )

    DGC_g = {'h1_0': ['a'], 'h0_0': ['v'], 'h2_0': ['ab']}

    f, integrate = generate_f_integral(DGC, DGC_g)

    print "DGC Toy"
    print "f(v) = ", f(['v'])
    print "f(a) = ", f(['a'])
    print "f(b) = ", f(['b'])
    print "f(aa) = ", f(['aa'])
    print "f(ab) = ", f(['ab'])
    print "f(a + b) = ", f(['a', 'b'])
    print "f(aa + ab) = ", f(['aa', 'ab'])

    print "\nintegrate(a x b) =", integrate(('a','b'))

    print "\nfacet_to_cells(b) =", facet_to_cells('b', DGC)

    # test data linked
    LNK = Coalgebra(
        {0: ['v'], 1: ['a', 'b'], 2: ['s', 't_{1}', 't_{2}'], 3: ['p', 'q']},
        {'q': {'s': 1, 't_{2}': 1, 't_{1}': 1}, 'p': {'s': 1}},
        {} # don't really care about coproduct definition
    )

    LNK_g = {'h1_0': ['a'], 'h0_0': ['v'], 'h2_0': ['t_{1}'], 'h1_1': ['b']}

    f, integrate = generate_f_integral(LNK, LNK_g)

    print "\n\nLINKED"
    print "f(v) = ", f(['v'])
    print "f(a) = ", f(['a'])
    print "f(b) = ", f(['b'])
    print "f(s) = ", f(['s'])
    print "f(t_{1}) = ", f(['t_{1}'])
    print "f(t_{2}) = ", f(['t_{2}'])
    print "f(p) = ", f(['p'])
    print "f(q) = ", f(['q'])

    print "\nfacet_to_cells(s) =", facet_to_cells('s', LNK)

    BR_C = Coalgebra(
        {0: ['v_{1}', 'v_{2}', 'v_{3}', 'v_{4}', 'v_{5}', 'v_{6}', 'v_{7}', 'v_{8}', 'v_{9}', 'v_{10}', 'v_{11}'], 1: ['m_{1}', 'm_{2}', 'm_{3}', 'm_{4}', 'm_{5}', 'm_{6}', 'm_{7}', 'm_{8}', 'm_{9}', 'm_{10}', 'm_{11}', 'm_{12}', 'm_{13}', 'm_{14}', 'c_{1}', 'c_{2}', 'c_{3}', 'c_{4}', 'c_{5}', 'c_{6}', 'c_{7}', 'c_{8}', 'c_{9}', 'c_{10}', 'c_{11}', 'c_{12}', 'c_{13}', 'c_{14}', 'c_{15}', 'c_{16}', 'c_{17}', 'c_{18}'], 2: ['a_{1}', 'a_{2}', 'a_{3}', 'a_{4}', 'e_{1}', 'e_{2}', 's_{1}', 's_{2}', 's_{3}', 's_{4}', 's_{5}', 's_{6}', 's_{7}', 's_{8}', 's_{9}', 's_{10}', 's_{11}', 's_{12}', 't_{1}', 't_{2}', 't_{3}', 't_{4}', 't_{5}', 't_{6}', 't_{7}', 't_{8}'], 3: ['D', 'q_{1}', 'q_{2}', 'q_{3}', 'q_{4}']},
        {'m_{1}': {'v_{11}': 1, 'v_{1}': 1}, 'c_{5}': {'v_{7}': 1, 'v_{8}': 1}, 'm_{3}': {'v_{11}': 1, 'v_{8}': 1}, 'c_{13}': {'v_{11}': 1, 'v_{10}': 1}, 'c_{7}': {'v_{2}': 1, 'v_{3}': 1}, 'c_{3}': {'v_{2}': 1, 'v_{3}': 1}, 't_{6}': {'m_{7}': 1, 'c_{2}': 1, 'm_{6}': 1, 'c_{5}': 1, 'c_{6}': 1}, 't_{2}': {'c_{3}': 1, 'm_{5}': 1, 'c_{4}': 1, 'c_{1}': 1, 'm_{4}': 1}, 's_{4}': {'m_{9}': 1, 'c_{16}': 1, 'm_{12}': 1, 'c_{9}': 1}, 'c_{1}': {'v_{5}': 1, 'v_{1}': 1}, 'm_{9}': {'v_{7}': 1, 'v_{6}': 1}, 't_{4}': {'c_{7}': 1, 'c_{8}': 1, 'm_{5}': 1, 'c_{11}': 1, 'm_{4}': 1}, 's_{6}': {'c_{10}': 1, 'm_{7}': 1, 'c_{14}': 1, 'm_{3}': 1}, 'e_{2}': {'c_{18}': 1, 'c_{16}': 1, 'c_{14}': 1, 'c_{11}': 1, 'c_{12}': 1}, 'a_{3}': {'m_{1}': 1, 'c_{17}': 1}, 't_{7}': {'m_{9}': 1, 'c_{10}': 1, 'm_{8}': 1, 'c_{9}': 1, 'c_{12}': 1}, 'm_{12}': {'v_{5}': 1, 'v_{8}': 1}, 'c_{11}': {'v_{5}': 1, 'v_{1}': 1}, 'c_{15}': {'v_{5}': 1, 'v_{6}': 1}, 'a_{1}': {'c_{17}': 1, 'm_{14}': 1}, 'q_{3}': {'a_{3}': 1, 's_{1}': 1, 'e_{1}': 1, 's_{5}': 1, 't_{2}': 1, 's_{11}': 1, 't_{6}': 1}, 'c_{9}': {'v_{7}': 1, 'v_{8}': 1}, 't_{8}': {'c_{10}': 1, 'm_{7}': 1, 'm_{6}': 1, 'c_{9}': 1, 'c_{12}': 1}, 'c_{17}': {'v_{11}': 1, 'v_{1}': 1}, 's_{8}': {'c_{8}': 1, 'm_{13}': 1, 'm_{10}': 1, 'm_{12}': 1, 'c_{10}': 1}, 'c_{16}': {'v_{5}': 1, 'v_{6}': 1}, 'm_{10}': {'v_{4}': 1, 'v_{5}': 1}, 's_{11}': {'m_{2}': 1, 'c_{15}': 1, 'm_{5}': 1, 'c_{4}': 1}, 'm_{2}': {'v_{3}': 1, 'v_{6}': 1}, 'm_{14}': {'v_{11}': 1, 'v_{1}': 1}, 'm_{6}': {'v_{7}': 1, 'v_{6}': 1}, 'q_{2}': {'e_{2}': 1, 'a_{2}': 1, 's_{10}': 1, 's_{8}': 1, 's_{4}': 1, 't_{3}': 1, 't_{7}': 1}, 's_{3}': {'m_{9}': 1, 'c_{5}': 1, 'c_{15}': 1, 'm_{12}': 1}, 'q_{4}': {'e_{2}': 1, 's_{2}': 1, 't_{8}': 1, 'a_{4}': 1, 's_{6}': 1, 't_{4}': 1, 's_{12}': 1}, 'D': {'a_{3}': 1, 'a_{4}': 1, 'a_{1}': 1, 'a_{2}': 1}, 'c_{10}': {'v_{8}': 1, 'v_{9}': 1}, 't_{1}': {'m_{11}': 1, 'c_{3}': 1, 'm_{10}': 1, 'c_{4}': 1, 'c_{1}': 1}, 'c_{6}': {'v_{8}': 1, 'v_{9}': 1}, 't_{5}': {'m_{9}': 1, 'c_{5}': 1, 'm_{8}': 1, 'c_{2}': 1, 'c_{6}': 1}, 's_{5}': {'m_{7}': 1, 'c_{13}': 1, 'm_{3}': 1, 'c_{6}': 1}, 's_{2}': {'m_{1}': 1, 'm_{3}': 1, 'c_{7}': 1, 'c_{9}': 1, 'm_{6}': 1, 'm_{2}': 1, 'm_{4}': 1}, 't_{3}': {'m_{11}': 1, 'c_{8}': 1, 'm_{10}': 1, 'c_{7}': 1, 'c_{11}': 1}, 'm_{13}': {'v_{3}': 1, 'v_{9}': 1}, 'm_{8}': {'v_{10}': 1, 'v_{9}': 1}, 's_{7}': {'m_{13}': 1, 'm_{10}': 1, 'm_{12}': 1, 'c_{6}': 1, 'c_{4}': 1}, 'm_{11}': {'v_{2}': 1, 'v_{1}': 1}, 'c_{8}': {'v_{4}': 1, 'v_{3}': 1}, 'c_{4}': {'v_{4}': 1, 'v_{3}': 1}, 'a_{2}': {'c_{18}': 1, 'm_{14}': 1}, 's_{1}': {'m_{1}': 1, 'c_{5}': 1, 'm_{3}': 1, 'c_{3}': 1, 'm_{6}': 1, 'm_{2}': 1, 'm_{4}': 1}, 'e_{1}': {'c_{17}': 1, 'c_{15}': 1, 'c_{2}': 1, 'c_{13}': 1, 'c_{1}': 1}, 'q_{1}': {'s_{3}': 1, 'e_{1}': 1, 'a_{1}': 1, 't_{1}': 1, 't_{5}': 1, 's_{7}': 1, 's_{9}': 1}, 'c_{2}': {'v_{10}': 1, 'v_{6}': 1}, 'c_{12}': {'v_{10}': 1, 'v_{6}': 1}, 's_{10}': {'m_{11}': 1, 'c_{7}': 1, 'c_{14}': 1, 'm_{13}': 1, 'm_{8}': 1, 'm_{14}': 1}, 'm_{7}': {'v_{10}': 1, 'v_{9}': 1}, 'a_{4}': {'m_{1}': 1, 'c_{18}': 1}, 'c_{14}': {'v_{11}': 1, 'v_{10}': 1}, 'c_{18}': {'v_{11}': 1, 'v_{1}': 1}, 'm_{4}': {'v_{2}': 1, 'v_{1}': 1}, 'm_{5}': {'v_{4}': 1, 'v_{5}': 1}, 's_{9}': {'m_{11}': 1, 'c_{13}': 1, 'c_{3}': 1, 'm_{13}': 1, 'm_{8}': 1, 'm_{14}': 1}, 's_{12}': {'m_{2}': 1, 'c_{16}': 1, 'm_{5}': 1, 'c_{8}': 1}},
        {} # don't really care about coproduct definition
    )

    BR_g = {'h0_0': ['v_{1}'], 'h2_1': ['t_{5}', 't_{6}', 't_{7}', 't_{8}'], 'h2_0': ['t_{1}', 't_{2}', 't_{3}', 't_{4}'], 'h1_0': ['m_{11}', 'm_{4}'], 'h1_1': ['c_{3}', 'c_{7}'], 'h1_2': ['m_{6}', 'm_{9}']}

    f_BR, integrate_BR = generate_f_integral(BR_C, BR_g)

    print "\n\nBorromean Rings"
    for c in [c for cells in BR_C.groups.values() for c in cells]:
        result_f_c = f_BR([c])
        if result_f_c:
            print "f(", c, ") = ", result_f_c

    for (l, r) in combinations(BR_C.groups[1], 2):
        result_f_cxc = f_BR([l, r])
        if result_f_cxc:
            print "f({} + {}) = {}".format(l, r, result_f_cxc)

    print "f(m_{4} + m_{11} + m_{8}) = ", f_BR(['m_{4}', 'm_{11}', 'm_{8}'])
    print "f(m_{4} + m_{11} + m_{14}) = ", f_BR(['m_{4}', 'm_{11}', 'm_{14}'])

    print "d( m11 ) = ", derivative('m_{11}', BR_C)
    print "d( (m_4, m_11) ) = ", derivative(('m_{4}', 'm_{11}'), BR_C)
    print "d( (m_4, m_11, v10) ) = ", derivative(('m_{4}', 'm_{11}', 'v_{10}'), BR_C)

    print "integrate((m_{4} + m{11}) x (v_{1} + v_{2})) = ", integrate_BR([(['m_{11}', 'm_{4}'], ['v_{2}', 'v_{1}'])])

    print "f(c_{3} + c_{7} + m_{4} + m_{11}) =", f_BR(['c_{3}', 'c_{7}', 'm_{4}', 'm_{11}'])

    BR_nabla_delta3 = {'m_{1}': [], 'v_{7}': [], 'c_{5}': [], 'm_{3}': [], 'v_{1}': [], 'v_{8}': [], 'v_{5}': [], 'c_{7}': [], 'c_{3}': [], 'v_{11}': [], 'a_{1}': [], 't_{2}': [(['v_{1}', 'v_{2}'], 'c_{3}', ['c_{4}', 'm_{5}']), (['v_{1}', 'v_{3}'], 'c_{4}', 'm_{5}'), (['c_{3}', 'm_{4}'], 'v_{4}', 'm_{5}'), (['c_{3}', 'm_{4}'], 'c_{4}', ['v_{4}', 'v_{5}']), ('m_{4}', ['v_{2}', 'v_{3}'], 'c_{4}'), ('m_{4}', 'c_{3}', ['v_{3}', 'v_{5}']), ('m_{4}', 'v_{2}', 'm_{5}'), ('c_{3}', 'v_{3}', 'm_{5}')], 's_{4}': [('c_{16}', 'm_{9}', ['v_{7}', 'v_{8}']), ('c_{16}', ['v_{6}', 'v_{7}'], 'c_{9}'), (['v_{5}', 'v_{6}'], 'm_{9}', 'c_{9}')], 'c_{1}': [], 'm_{9}': [], 'v_{9}': [], 't_{4}': [(['c_{7}', 'm_{4}'], 'v_{4}', 'm_{5}'), ('m_{4}', 'c_{7}', ['v_{3}', 'v_{5}']), (['c_{7}', 'm_{4}'], 'c_{8}', ['v_{4}', 'v_{5}']), (['v_{1}', 'v_{2}'], 'c_{7}', ['c_{8}', 'm_{5}']), (['v_{1}', 'v_{3}'], 'c_{8}', 'm_{5}'), ('m_{4}', ['v_{2}', 'v_{3}'], 'c_{8}'), ('m_{4}', 'v_{2}', 'm_{5}'), ('c_{7}', 'v_{3}', 'm_{5}')], 's_{6}': [('c_{10}', 'm_{7}', ['v_{10}', 'v_{11}']), ('c_{10}', ['v_{10}', 'v_{9}'], 'c_{14}'), (['v_{8}', 'v_{9}'], 'm_{7}', 'c_{14}')], 't_{6}': [('m_{6}', ['v_{7}', 'v_{8}'], 'c_{6}'), (['c_{5}', 'm_{6}'], 'c_{6}', ['v_{10}', 'v_{9}']), (['c_{5}', 'm_{6}'], 'v_{9}', 'm_{7}'), ('m_{6}', 'c_{5}', ['v_{10}', 'v_{8}']), ('v_{6}', ['c_{5}', 'c_{6}'], 'm_{7}'), (['v_{6}', 'v_{7}'], 'c_{5}', 'c_{6}'), ('v_{7}', 'c_{5}', 'm_{7}'), ('c_{5}', 'v_{8}', 'm_{7}'), ('v_{8}', 'c_{6}', 'm_{7}'), ('m_{6}', 'v_{7}', 'm_{7}')], 'e_{2}': [('c_{11}', 'c_{16}', ['v_{11}', 'v_{6}']), ('v_{1}', ['c_{12}', 'c_{16}'], 'c_{14}'), (['v_{1}', 'v_{5}'], 'c_{16}', 'c_{12}'), (['c_{11}', 'c_{16}'], 'v_{10}', 'c_{14}'), (['c_{11}', 'c_{16}'], 'c_{12}', ['v_{10}', 'v_{11}']), ('c_{11}', ['v_{5}', 'v_{6}'], 'c_{12}'), ('v_{6}', 'c_{12}', 'c_{14}'), ('c_{11}', 'v_{5}', 'c_{14}'), ('v_{5}', 'c_{16}', 'c_{14}'), ('c_{16}', 'v_{6}', 'c_{14}')], 'a_{3}': [], 's_{2}': [(['c_{7}', 'm_{2}', 'm_{4}'], 'm_{6}', ['v_{11}', 'v_{7}']), (['c_{7}', 'm_{4}'], 'v_{6}', 'm_{6}'), (['v_{1}', 'v_{2}'], 'c_{7}', ['c_{9}', 'm_{2}', 'm_{3}', 'm_{6}']), ('v_{1}', ['c_{9}', 'm_{2}', 'm_{6}'], 'm_{3}'), ('v_{1}', ['m_{2}', 'm_{6}'], 'c_{9}'), (['v_{1}', 'v_{3}'], 'm_{2}', 'm_{6}'), (['c_{7}', 'm_{2}', 'm_{4}', 'm_{6}'], 'v_{8}', 'm_{3}'), (['c_{7}', 'm_{2}', 'm_{4}', 'm_{6}'], 'c_{9}', ['v_{11}', 'v_{8}']), (['c_{7}', 'm_{2}', 'm_{4}'], 'v_{7}', 'c_{9}'), (['c_{7}', 'm_{4}'], 'm_{2}', ['v_{11}', 'v_{6}']), ('m_{4}', 'c_{7}', ['v_{11}', 'v_{3}']), ('m_{4}', ['v_{2}', 'v_{3}'], 'm_{2}'), ('c_{7}', 'v_{3}', ['c_{9}', 'm_{3}', 'm_{6}']), ('m_{4}', 'v_{2}', ['c_{9}', 'm_{3}', 'm_{6}']), ('v_{3}', 'm_{2}', ['c_{9}', 'm_{3}']), ('m_{2}', 'v_{6}', ['c_{9}', 'm_{3}']), ('v_{7}', 'c_{9}', 'm_{3}'), ('v_{6}', 'm_{6}', ['c_{9}', 'm_{3}']), ('m_{6}', 'v_{7}', 'm_{3}')], 'm_{12}': [], 'c_{11}': [], 'c_{15}': [], 'c_{13}': [], 'q_{3}': [(['s_{11}', 't_{2}'], 'v_{7}', 'c_{5}'), (['s_{11}', 't_{2}'], 'c_{5}', ['v_{11}', 'v_{8}']), (['c_{15}', 'c_{1}'], 'm_{6}', ['c_{5}', 'c_{6}', 'm_{7}']), (['c_{15}', 'c_{1}'], ['c_{5}', 'c_{6}', 'c_{6}'], 'm_{7}'), ('c_{1}', ['v_{5}', 'v_{6}'], 't_{6}'), (['c_{15}', 'c_{1}'], 'c_{5}', 'c_{6}'), (['c_{15}', 'c_{1}'], 't_{6}', ['v_{10}', 'v_{11}']), (['s_{11}', 't_{2}'], 'm_{6}', ['v_{11}', 'v_{7}']), ('t_{2}', ['v_{5}', 'v_{6}'], 'm_{6}'), (['s_{11}', 't_{2}'], 'v_{8}', 'm_{3}'), (['c_{15}', 'c_{1}'], ['c_{6}', 'm_{7}'], 'c_{13}'), (['c_{15}', 'c_{1}', 'm_{6}'], 'v_{8}', 's_{5}'), ('t_{2}', 'c_{15}', ['v_{11}', 'v_{6}']), ('m_{4}', ['v_{2}', 'v_{3}'], 's_{11}'), (['c_{3}', 'm_{4}'], 'c_{4}', 'm_{5}'), (['c_{3}', 'm_{4}'], 's_{11}', ['v_{11}', 'v_{6}']), (['v_{1}', 'v_{3}'], 's_{11}', ['c_{5}', 'm_{3}', 'm_{6}']), (['v_{1}', 'v_{5}'], 'c_{15}', ['s_{5}', 't_{6}']), (['v_{1}', 'v_{6}'], 't_{6}', 'c_{13}'), ('v_{1}', ['c_{5}', 'm_{6}'], 's_{5}'), (['v_{1}', 'v_{2}'], 'c_{3}', 's_{11}'), (['c_{5}', 'm_{6}'], 'c_{6}', 'm_{7}'), (['c_{4}', 'm_{5}'], 'c_{15}', ['c_{5}', 'm_{3}', 'm_{6}']), ('s_{11}', 'v_{6}', ['c_{5}', 'm_{3}']), ('m_{4}', ['c_{3}', 'c_{4}', 'm_{5}'], ['c_{5}', 'm_{6}']), ('c_{3}', ['c_{4}', 'm_{5}'], ['c_{5}', 'm_{3}', 'm_{6}']), ('t_{2}', 'v_{5}', ['c_{5}', 'm_{3}']), ('c_{15}', 'v_{6}', 's_{5}'), ('m_{4}', 'c_{3}', ['c_{15}', 'm_{3}']), ('c_{4}', 'm_{5}', 'c_{15}'), ('c_{1}', 'v_{5}', 's_{5}'), ('m_{4}', ['c_{4}', 'm_{5}'], 'm_{3}'), ('m_{6}', 'c_{5}', 'c_{13}'), ('c_{6}', 'm_{7}', 'c_{13}'), ('v_{7}', 'c_{5}', 's_{5}'), ('m_{6}', 'v_{7}', 's_{5}'), ('v_{6}', 'm_{6}', 's_{5}')], 'c_{9}': [], 't_{8}': [('m_{6}', 'c_{9}', ['v_{10}', 'v_{8}']), ('m_{6}', ['v_{7}', 'v_{8}'], 'c_{10}'), (['c_{9}', 'm_{6}'], 'c_{10}', ['v_{10}', 'v_{9}']), ('v_{6}', ['c_{10}', 'c_{9}'], 'm_{7}'), (['v_{6}', 'v_{7}'], 'c_{9}', 'c_{10}'), (['c_{9}', 'm_{6}'], 'v_{9}', 'm_{7}'), ('v_{8}', 'c_{10}', 'm_{7}'), ('c_{9}', 'v_{8}', 'm_{7}'), ('v_{7}', 'c_{9}', 'm_{7}'), ('m_{6}', 'v_{7}', 'm_{7}')], 'c_{17}': [], 's_{8}': [('v_{3}', ['m_{10}', 'm_{12}'], 'c_{10}'), (['v_{3}', 'v_{4}'], 'm_{10}', 'm_{12}'), ('c_{8}', 'm_{10}', ['v_{5}', 'v_{9}']), (['c_{8}', 'm_{10}'], 'v_{8}', 'c_{10}'), (['c_{8}', 'm_{10}'], 'm_{12}', ['v_{8}', 'v_{9}']), ('c_{8}', ['v_{4}', 'v_{5}'], 'm_{12}'), ('v_{4}', 'm_{10}', 'c_{10}'), ('m_{10}', 'v_{5}', 'c_{10}'), ('v_{5}', 'm_{12}', 'c_{10}'), ('c_{8}', 'v_{4}', 'c_{10}')], 'c_{16}': [], 'm_{6}': [], 's_{11}': [(['v_{3}', 'v_{4}'], 'm_{5}', 'c_{15}'), ('c_{4}', 'm_{5}', ['v_{5}', 'v_{6}']), ('c_{4}', ['v_{4}', 'v_{5}'], 'c_{15}')], 'm_{2}': [], 'v_{2}': [], 'm_{14}': [], 'q_{2}': [('v_{1}', ['s_{4}', 's_{8}', 't_{7}'], 'c_{14}'), ('v_{1}', 's_{4}', ['c_{10}', 'm_{8}']), (['v_{1}', 'v_{5}'], 'c_{16}', 't_{7}'), (['v_{1}', 'v_{2}'], 'c_{7}', 's_{8}'), (['v_{1}', 'v_{3}'], 's_{8}', 'm_{8}'), (['s_{4}', 's_{8}', 't_{3}'], 'v_{10}', 'c_{14}'), ('t_{3}', 'm_{12}', ['v_{11}', 'v_{8}']), (['s_{4}', 't_{3}'], 'c_{10}', ['v_{11}', 'v_{9}']), ('t_{3}', ['v_{5}', 'v_{8}'], 'c_{10}'), (['c_{11}', 'c_{16}', 'c_{8}', 'm_{10}', 'm_{12}'], 'c_{10}', 'm_{8}'), (['c_{11}', 'c_{11}', 'c_{16}'], 'm_{9}', 'c_{9}'), (['c_{11}', 'c_{16}'], 't_{7}', ['v_{10}', 'v_{11}']), ('c_{11}', 'c_{16}', ['c_{9}', 'm_{9}']), ('c_{11}', 's_{4}', ['v_{11}', 'v_{8}']), ('m_{11}', ['v_{2}', 'v_{3}'], 's_{8}'), (['c_{7}', 'm_{11}'], 'c_{8}', ['c_{14}', 'm_{10}', 'm_{8}']), (['c_{7}', 'm_{11}'], 'm_{12}', 'c_{10}'), (['c_{7}', 'm_{11}'], 's_{8}', ['v_{11}', 'v_{9}']), ('c_{11}', ['c_{9}', 'm_{9}'], ['c_{10}', 'm_{8}']), ('c_{11}', ['v_{5}', 'v_{6}'], 't_{7}'), (['s_{4}', 't_{3}'], 'v_{9}', 'm_{8}'), (['s_{4}', 's_{8}', 't_{3}'], 'm_{8}', ['v_{10}', 'v_{11}']), ('c_{16}', ['c_{9}', 'm_{9}'], 'c_{14}'), ('s_{4}', 'v_{8}', ['c_{14}', 'm_{8}']), ('v_{5}', 's_{4}', ['c_{10}', 'c_{14}', 'm_{8}']), ('c_{8}', 'm_{10}', ['c_{10}', 'm_{12}']), ('m_{11}', 'c_{7}', ['c_{10}', 'c_{14}', 'm_{12}', 'm_{8}']), (['c_{10}', 'c_{9}', 'm_{9}'], 'm_{8}', 'c_{14}'), ('v_{6}', 't_{7}', 'c_{14}'), (['c_{8}', 'c_{9}', 'm_{10}', 'm_{12}', 'm_{9}'], 'c_{10}', 'c_{14}'), ('m_{9}', 'c_{9}', ['c_{10}', 'm_{8}']), ('v_{3}', 's_{8}', 'c_{14}'), (['c_{8}', 'm_{10}'], 'm_{12}', ['c_{14}', 'm_{8}']), ('s_{8}', 'v_{9}', 'c_{14}'), ('t_{3}', 'v_{5}', ['c_{14}', 'm_{8}']), (['c_{7}', 'm_{11}'], 'm_{10}', ['c_{14}', 'm_{8}'])], 's_{3}': [(['v_{5}', 'v_{6}'], 'm_{9}', 'c_{5}'), ('c_{15}', 'm_{9}', ['v_{7}', 'v_{8}']), ('c_{15}', ['v_{6}', 'v_{7}'], 'c_{5}')], 'q_{4}': [(['v_{1}', 'v_{5}'], 'c_{16}', ['s_{6}', 't_{8}']), (['v_{1}', 'v_{3}'], 's_{12}', ['c_{9}', 'm_{3}', 'm_{6}']), (['v_{1}', 'v_{2}'], 'c_{7}', 's_{12}'), ('v_{1}', ['c_{9}', 'm_{6}'], 's_{6}'), (['v_{1}', 'v_{6}'], 't_{8}', 'c_{14}'), (['c_{11}', 'c_{16}'], 'm_{6}', ['c_{10}', 'c_{9}', 'm_{7}']), (['c_{11}', 'c_{16}'], 't_{8}', ['v_{10}', 'v_{11}']), (['c_{11}', 'c_{16}'], ['c_{10}', 'c_{10}', 'c_{9}'], 'm_{7}'), (['c_{11}', 'c_{16}'], 'c_{9}', 'c_{10}'), ('c_{11}', ['v_{5}', 'v_{6}'], 't_{8}'), (['c_{11}', 'c_{16}'], ['c_{10}', 'm_{7}'], 'c_{14}'), (['c_{11}', 'c_{16}', 'm_{6}'], 'v_{8}', 's_{6}'), (['s_{12}', 't_{4}'], 'm_{6}', 'v_{7}'), ('t_{4}', ['v_{5}', 'v_{6}'], 'm_{6}'), (['s_{12}', 't_{4}'], 'c_{9}', 'v_{8}'), (['s_{12}', 't_{4}'], 'v_{7}', 'c_{9}'), (['s_{12}', 't_{4}'], 'v_{8}', 'm_{3}'), (['c_{7}', 'm_{4}'], 's_{12}', ['v_{11}', 'v_{6}']), (['c_{7}', 'm_{4}'], 'c_{8}', ['m_{3}', 'm_{5}']), (['c_{9}', 'm_{6}'], 'c_{10}', 'm_{7}'), ('t_{4}', 'c_{16}', ['v_{11}', 'v_{6}']), ('m_{4}', ['v_{2}', 'v_{3}'], 's_{12}'), ('c_{16}', 'v_{6}', 's_{6}'), ('m_{4}', ['c_{7}', 'c_{8}', 'm_{5}'], ['c_{9}', 'm_{6}']), ('c_{7}', ['c_{8}', 'm_{5}'], ['c_{9}', 'm_{6}']), ('t_{4}', 'v_{5}', ['c_{9}', 'm_{3}']), ('s_{12}', 'v_{6}', ['c_{9}', 'm_{3}']), (['c_{8}', 'm_{5}'], 'c_{16}', ['c_{9}', 'm_{3}', 'm_{6}']), ('c_{11}', 'v_{5}', 's_{6}'), ('m_{6}', 'v_{7}', 's_{6}'), ('v_{6}', 'm_{6}', 's_{6}'), ('v_{7}', 'c_{9}', 's_{6}'), ('m_{4}', 'c_{7}', ['c_{16}', 'm_{3}']), ('c_{8}', 'm_{5}', 'c_{16}'), ('m_{6}', 'c_{9}', 'c_{14}'), ('c_{10}', 'm_{7}', 'c_{14}'), (['s_{12}', 't_{4}'], ['c_{9}', 'm_{6}'], 'v_{11}'), (['c_{7}', 'm_{4}'], 'm_{5}', 'm_{3}')], 'D': [], 'm_{8}': [], 'c_{8}': [], 'v_{4}': [], 't_{1}': [(['c_{3}', 'm_{11}'], 'c_{4}', ['v_{4}', 'v_{5}']), (['c_{3}', 'm_{11}'], 'v_{4}', 'm_{10}'), (['v_{1}', 'v_{2}'], 'c_{3}', ['c_{4}', 'm_{10}']), (['v_{1}', 'v_{3}'], 'c_{4}', 'm_{10}'), ('m_{11}', 'c_{3}', ['v_{3}', 'v_{5}']), ('m_{11}', ['v_{2}', 'v_{3}'], 'c_{4}'), ('c_{3}', 'v_{3}', 'm_{10}'), ('m_{11}', 'v_{2}', 'm_{10}')], 'v_{6}': [], 'c_{6}': [], 't_{5}': [('v_{6}', ['c_{5}', 'c_{6}'], 'm_{8}'), (['v_{6}', 'v_{7}'], 'c_{5}', 'c_{6}'), (['c_{5}', 'm_{9}'], 'v_{9}', 'm_{8}'), ('m_{9}', ['v_{7}', 'v_{8}'], 'c_{6}'), (['c_{5}', 'm_{9}'], 'c_{6}', ['v_{10}', 'v_{9}']), ('m_{9}', 'c_{5}', ['v_{10}', 'v_{8}']), ('v_{8}', 'c_{6}', 'm_{8}'), ('m_{9}', 'v_{7}', 'm_{8}'), ('v_{7}', 'c_{5}', 'm_{8}'), ('c_{5}', 'v_{8}', 'm_{8}')], 's_{5}': [('c_{6}', ['v_{10}', 'v_{9}'], 'c_{13}'), (['v_{8}', 'v_{9}'], 'm_{7}', 'c_{13}'), ('c_{6}', 'm_{7}', ['v_{10}', 'v_{11}'])], 'm_{10}': [], 'v_{10}': [], 't_{3}': [('m_{11}', ['v_{2}', 'v_{3}'], 'c_{8}'), (['c_{7}', 'm_{11}'], 'c_{8}', ['v_{4}', 'v_{5}']), ('m_{11}', 'c_{7}', ['v_{3}', 'v_{5}']), (['c_{7}', 'm_{11}'], 'v_{4}', 'm_{10}'), ('v_{1}', ['c_{7}', 'c_{8}'], 'm_{10}'), (['v_{1}', 'v_{2}'], 'c_{7}', 'c_{8}'), ('v_{3}', 'c_{8}', 'm_{10}'), ('m_{11}', 'v_{2}', 'm_{10}'), ('c_{7}', 'v_{3}', 'm_{10}'), ('v_{2}', 'c_{7}', 'm_{10}')], 'm_{13}': [], 't_{7}': [(['c_{9}', 'm_{9}'], 'v_{9}', 'm_{8}'), ('v_{6}', ['c_{10}', 'c_{9}'], 'm_{8}'), (['v_{6}', 'v_{7}'], 'c_{9}', 'c_{10}'), ('m_{9}', 'c_{9}', ['v_{10}', 'v_{8}']), (['c_{9}', 'm_{9}'], 'c_{10}', ['v_{10}', 'v_{9}']), ('m_{9}', ['v_{7}', 'v_{8}'], 'c_{10}'), ('c_{9}', 'v_{8}', 'm_{8}'), ('v_{7}', 'c_{9}', 'm_{8}'), ('m_{9}', 'v_{7}', 'm_{8}'), ('v_{8}', 'c_{10}', 'm_{8}')], 's_{7}': [(['c_{4}', 'm_{10}'], 'v_{8}', 'c_{6}'), ('v_{3}', ['m_{10}', 'm_{12}'], 'c_{6}'), (['v_{3}', 'v_{4}'], 'm_{10}', 'm_{12}'), (['c_{4}', 'm_{10}'], 'm_{12}', ['v_{8}', 'v_{9}']), ('c_{4}', ['v_{4}', 'v_{5}'], 'm_{12}'), ('c_{4}', 'm_{10}', ['v_{5}', 'v_{9}']), ('v_{4}', 'm_{10}', 'c_{6}'), ('m_{10}', 'v_{5}', 'c_{6}'), ('v_{5}', 'm_{12}', 'c_{6}'), ('c_{4}', 'v_{4}', 'c_{6}')], 'm_{11}': [], 'c_{10}': [], 'c_{4}': [], 'a_{2}': [], 's_{1}': [(['c_{3}', 'm_{2}', 'm_{4}', 'm_{6}'], 'v_{8}', 'm_{3}'), ('v_{1}', ['c_{3}', 'c_{5}', 'm_{2}', 'm_{6}'], 'm_{3}'), ('v_{1}', ['c_{3}', 'm_{2}', 'm_{6}'], 'c_{5}'), ('v_{1}', 'c_{3}', ['m_{2}', 'm_{6}']), (['v_{1}', 'v_{3}'], 'm_{2}', 'm_{6}'), ('m_{4}', 'c_{3}', ['v_{11}', 'v_{3}']), (['c_{3}', 'm_{2}', 'm_{4}'], 'v_{7}', 'c_{5}'), (['c_{3}', 'm_{2}', 'm_{4}', 'm_{6}'], 'c_{5}', ['v_{11}', 'v_{8}']), (['c_{3}', 'm_{2}', 'm_{4}'], 'm_{6}', 'v_{7}'), (['c_{3}', 'm_{4}'], 'v_{6}', 'm_{6}'), ('m_{4}', ['v_{2}', 'v_{3}'], 'm_{2}'), (['c_{3}', 'm_{4}'], 'm_{2}', 'v_{6}'), ('v_{2}', 'c_{3}', ['c_{5}', 'm_{2}', 'm_{3}', 'm_{6}']), ('c_{3}', 'v_{3}', ['c_{5}', 'm_{3}', 'm_{6}']), ('m_{4}', 'v_{2}', ['c_{5}', 'm_{3}', 'm_{6}']), ('v_{7}', 'c_{5}', 'm_{3}'), ('v_{3}', 'm_{2}', ['c_{5}', 'm_{3}']), ('m_{2}', 'v_{6}', ['c_{5}', 'm_{3}']), ('m_{6}', 'v_{7}', 'm_{3}'), ('v_{6}', 'm_{6}', ['c_{5}', 'm_{3}']), (['c_{3}', 'm_{4}'], ['m_{2}', 'm_{6}'], 'v_{11}'), ('m_{2}', 'm_{6}', 'v_{11}')], 'e_{1}': [(['c_{15}', 'c_{1}'], 'v_{10}', 'c_{13}'), (['c_{15}', 'c_{1}'], 'c_{2}', ['v_{10}', 'v_{11}']), ('c_{1}', ['v_{5}', 'v_{6}'], 'c_{2}'), ('c_{1}', 'c_{15}', ['v_{11}', 'v_{6}']), (['v_{1}', 'v_{5}'], 'c_{15}', ['c_{13}', 'c_{2}']), (['v_{1}', 'v_{6}'], 'c_{2}', 'c_{13}'), ('c_{1}', 'v_{5}', 'c_{13}'), ('c_{15}', 'v_{6}', 'c_{13}')], 'm_{5}': [], 'c_{2}': [], 'c_{12}': [], 's_{10}': [(['c_{7}', 'm_{11}', 'm_{13}'], 'v_{10}', 'c_{14}'), (['c_{7}', 'm_{11}'], 'v_{9}', 'm_{8}'), (['c_{7}', 'm_{11}', 'm_{13}'], 'm_{8}', 'v_{10}'), (['c_{7}', 'm_{11}'], 'm_{13}', 'v_{9}'), ('m_{11}', ['v_{2}', 'v_{3}'], 'm_{13}'), ('m_{11}', 'c_{7}', ['v_{11}', 'v_{3}']), (['v_{1}', 'v_{2}'], 'c_{7}', ['c_{14}', 'm_{13}', 'm_{8}']), ('v_{1}', ['m_{13}', 'm_{8}'], 'c_{14}'), (['v_{1}', 'v_{3}'], 'm_{13}', 'm_{8}'), ('c_{7}', 'v_{3}', ['c_{14}', 'm_{8}']), ('m_{11}', 'v_{2}', ['c_{14}', 'm_{8}']), ('v_{9}', 'm_{8}', 'c_{14}'), ('m_{13}', 'v_{9}', 'c_{14}'), ('v_{3}', 'm_{13}', 'c_{14}'), (['c_{7}', 'm_{11}'], ['m_{13}', 'm_{8}'], 'v_{11}'), ('m_{13}', 'm_{8}', 'v_{11}')], 'm_{7}': [], 'a_{4}': [], 'c_{14}': [], 'c_{18}': [], 'm_{4}': [], 'q_{1}': [(['s_{3}', 's_{7}', 't_{1}'], 'v_{10}', 'c_{13}'), (['s_{3}', 's_{7}', 't_{1}'], 'm_{8}', ['v_{10}', 'v_{11}']), (['s_{3}', 't_{1}'], 'c_{6}', ['v_{11}', 'v_{9}']), (['c_{15}', 'c_{1}', 'c_{4}', 'm_{10}', 'm_{12}'], 'c_{6}', 'm_{8}'), (['c_{15}', 'c_{1}'], 't_{5}', ['v_{10}', 'v_{11}']), (['c_{15}', 'c_{1}', 'c_{1}'], 'm_{9}', 'c_{5}'), ('c_{1}', ['v_{5}', 'v_{6}'], 't_{5}'), ('c_{1}', ['c_{5}', 'm_{9}'], ['c_{6}', 'm_{8}']), ('t_{1}', ['v_{5}', 'v_{8}'], 'c_{6}'), (['s_{3}', 't_{1}'], 'v_{9}', 'm_{8}'), ('t_{1}', 'm_{12}', ['v_{11}', 'v_{8}']), (['c_{3}', 'm_{11}'], 'm_{12}', 'c_{6}'), (['c_{3}', 'm_{11}'], 's_{7}', ['v_{11}', 'v_{9}']), (['c_{3}', 'm_{11}'], 'c_{4}', 'm_{10}'), ('v_{1}', ['s_{3}', 's_{7}'], 'm_{8}'), ('v_{1}', 's_{3}', ['c_{13}', 'c_{6}']), (['v_{1}', 'v_{5}'], 'c_{15}', 't_{5}'), ('v_{1}', ['s_{7}', 't_{5}'], 'c_{13}'), (['v_{1}', 'v_{2}'], 'c_{3}', 's_{7}'), ('c_{1}', 'c_{15}', ['c_{5}', 'm_{9}']), ('c_{1}', 's_{3}', ['v_{11}', 'v_{8}']), ('m_{11}', ['v_{2}', 'v_{3}'], 's_{7}'), ('c_{3}', ['c_{4}', 'm_{10}'], ['c_{13}', 'm_{8}']), ('m_{11}', ['c_{3}', 'c_{4}', 'm_{10}'], 'c_{13}'), ('t_{1}', 'v_{5}', ['c_{13}', 'm_{8}']), ('v_{3}', 's_{7}', ['c_{13}', 'm_{8}']), (['c_{4}', 'm_{10}'], 'm_{12}', ['c_{13}', 'm_{8}']), ('v_{5}', 's_{3}', ['c_{13}', 'c_{6}', 'm_{8}']), ('m_{9}', 'c_{5}', ['c_{6}', 'm_{8}']), ('m_{11}', 'c_{3}', ['c_{6}', 'm_{12}', 'm_{8}']), ('c_{4}', 'm_{10}', ['c_{6}', 'm_{12}']), ('s_{3}', 'v_{8}', ['c_{13}', 'm_{8}']), ('m_{11}', ['c_{4}', 'm_{10}'], 'm_{8}'), (['c_{4}', 'c_{5}', 'm_{10}', 'm_{12}', 'm_{9}'], 'c_{6}', 'c_{13}'), ('s_{7}', 'v_{9}', 'c_{13}'), (['c_{5}', 'c_{6}', 'm_{9}'], 'm_{8}', 'c_{13}'), ('v_{6}', 't_{5}', 'c_{13}'), ('c_{15}', ['c_{5}', 'm_{9}'], 'c_{13}')], 'v_{3}': [], 's_{9}': [(['c_{3}', 'm_{11}'], 'v_{9}', 'm_{8}'), (['c_{3}', 'm_{11}', 'm_{13}'], 'm_{8}', ['v_{10}', 'v_{11}']), (['c_{3}', 'm_{11}'], 'm_{13}', ['v_{11}', 'v_{9}']), ('m_{11}', ['v_{2}', 'v_{3}'], 'm_{13}'), (['c_{3}', 'm_{11}', 'm_{13}'], 'v_{10}', 'c_{13}'), ('v_{1}', ['c_{3}', 'm_{13}'], 'm_{8}'), ('v_{1}', ['c_{3}', 'm_{13}', 'm_{8}'], 'c_{13}'), (['v_{1}', 'v_{2}'], 'c_{3}', 'm_{13}'), ('m_{11}', 'c_{3}', ['v_{11}', 'v_{3}']), ('m_{11}', 'v_{2}', ['c_{13}', 'm_{8}']), ('v_{3}', 'm_{13}', ['c_{13}', 'm_{8}']), ('v_{9}', 'm_{8}', 'c_{13}'), ('v_{2}', 'c_{3}', ['c_{13}', 'm_{8}']), ('c_{3}', 'v_{3}', ['c_{13}', 'm_{8}']), ('m_{13}', 'v_{9}', 'c_{13}')], 's_{12}': [(['v_{3}', 'v_{4}'], 'm_{5}', 'c_{16}'), ('c_{8}', ['v_{4}', 'v_{5}'], 'c_{16}'), ('c_{8}', 'm_{5}', ['v_{5}', 'v_{6}'])]}
    #print "integrate(nabla Delta3) =", integrate_BR(BR_nabla_delta3)
    BR_nabal_delta3_rem = {'m_{1}': [], 'v_{7}': [], 'c_{5}': [], 'm_{3}': [], 'v_{1}': [], 'v_{8}': [], 'v_{5}': [], 'c_{7}': [], 'c_{3}': [], 'v_{11}': [], 't_{2}': [], 's_{4}': [], 'c_{1}': [], 'm_{9}': [], 'v_{9}': [], 't_{4}': [], 'm_{14}': [], 't_{6}': [], 'e_{2}': [], 'a_{3}': [], 'm_{10}': [], 'm_{12}': [], 'c_{11}': [], 'c_{15}': [], 'a_{1}': [], 't_{8}': [], 'c_{9}': [], 'c_{17}': [], 's_{8}': [], 'c_{16}': [], 'm_{6}': [], 's_{11}': [], 'm_{2}': [], 'v_{2}': [], 's_{6}': [], 'm_{4}': [], 's_{3}': [], 'q_{4}': [('m_{4}', 'c_{7}', 'm_{6}'), ('c_{10}', 'm_{7}', 'c_{14}'), ('m_{4}', 'c_{8}', 'm_{5}'), ('c_{9}', 'c_{10}', 'm_{7}'), ('m_{6}', 'c_{10}', 'm_{7}'), ('c_{7}', 'c_{8}', 'm_{5}'), ('c_{8}', 'm_{5}', 'c_{16}'), ('m_{4}', 'c_{7}', 'c_{9}'), ('m_{4}', 'c_{7}', 'm_{3}'), ('m_{6}', 'c_{9}', 'm_{3}'), ('c_{11}', 'c_{16}', 'c_{12}'), ('m_{4}', 'm_{2}', 'c_{9}'), ('c_{11}', 'c_{16}', 'c_{14}'), ('m_{4}', 'm_{6}', 'm_{3}'), ('m_{4}', 'm_{2}', 'm_{6}'), ('c_{7}', 'm_{6}', 'c_{9}'), ('c_{7}', 'm_{6}', 'm_{3}'), ('c_{7}', 'm_{2}', 'm_{6}'), ('m_{4}', 'c_{7}', 'c_{8}'), ('m_{6}', 'c_{9}', 'c_{10}'), ('m_{4}', 'c_{7}', 'm_{2}'), ('c_{16}', 'c_{12}', 'c_{14}'), ('c_{7}', 'c_{9}', 'm_{3}'), ('m_{4}', 'm_{6}', 'c_{9}'), ('m_{6}', 'c_{9}', 'm_{7}'), ('m_{2}', 'c_{9}', 'm_{3}'), ('c_{7}', 'm_{2}', 'm_{3}'), ('c_{7}', 'm_{2}', 'c_{9}'), ('c_{11}', 'c_{12}', 'c_{14}'), ('m_{4}', 'c_{9}', 'm_{3}'), ('m_{4}', 'c_{7}', 'm_{5}'), ('m_{4}', 'm_{2}', 'm_{3}'), ('m_{2}', 'm_{6}', 'm_{3}'), ('m_{2}', 'm_{6}', 'c_{9}')], 't_{7}': [], 'D': [], 'c_{8}': [], 'v_{4}': [], 'c_{13}': [], 't_{1}': [], 'v_{6}': [], 'c_{6}': [], 't_{5}': [], 's_{5}': [], 's_{2}': [], 'v_{10}': [], 't_{3}': [], 'm_{13}': [], 'm_{8}': [], 's_{7}': [], 'm_{11}': [], 'c_{10}': [], 'c_{4}': [], 'a_{2}': [], 's_{1}': [], 'e_{1}': [], 'q_{1}': [('m_{11}', 'c_{4}', 'm_{10}'), ('c_{15}', 'm_{9}', 'c_{5}'), ('c_{4}', 'm_{10}', 'm_{12}'), ('m_{9}', 'c_{5}', 'c_{6}'), ('m_{11}', 'c_{3}', 'm_{8}'), ('m_{11}', 'c_{3}', 'c_{13}'), ('c_{3}', 'c_{4}', 'm_{10}'), ('m_{9}', 'c_{5}', 'm_{8}'), ('c_{4}', 'm_{10}', 'c_{6}'), ('c_{15}', 'c_{2}', 'c_{13}'), ('c_{3}', 'm_{13}', 'c_{13}'), ('c_{1}', 'c_{2}', 'c_{13}'), ('m_{13}', 'm_{8}', 'c_{13}'), ('m_{11}', 'c_{3}', 'c_{4}'), ('c_{4}', 'm_{12}', 'c_{6}'), ('m_{10}', 'm_{12}', 'c_{6}'), ('m_{11}', 'm_{13}', 'm_{8}'), ('m_{11}', 'c_{3}', 'm_{13}'), ('m_{9}', 'c_{6}', 'm_{8}'), ('c_{3}', 'm_{13}', 'm_{8}'), ('m_{11}', 'm_{13}', 'c_{13}'), ('m_{11}', 'c_{3}', 'm_{10}'), ('c_{3}', 'm_{8}', 'c_{13}'), ('m_{11}', 'm_{8}', 'c_{13}'), ('c_{1}', 'c_{15}', 'c_{2}'), ('c_{1}', 'c_{15}', 'c_{13}'), ('c_{5}', 'c_{6}', 'm_{8}')], 'q_{3}': [('c_{4}', 'm_{5}', 'c_{15}'), ('m_{6}', 'c_{6}', 'm_{7}'), ('c_{5}', 'c_{6}', 'm_{7}'), ('m_{4}', 'c_{3}', 'm_{6}'), ('c_{3}', 'c_{4}', 'm_{5}'), ('m_{4}', 'c_{3}', 'm_{3}'), ('m_{4}', 'c_{3}', 'c_{5}'), ('m_{4}', 'c_{4}', 'm_{5}'), ('c_{6}', 'm_{7}', 'c_{13}'), ('c_{3}', 'm_{6}', 'c_{5}'), ('m_{4}', 'c_{3}', 'm_{2}'), ('m_{6}', 'c_{5}', 'c_{6}'), ('c_{15}', 'c_{2}', 'c_{13}'), ('m_{4}', 'm_{2}', 'c_{5}'), ('c_{3}', 'm_{2}', 'c_{5}'), ('m_{4}', 'm_{6}', 'm_{3}'), ('m_{4}', 'm_{2}', 'm_{3}'), ('c_{3}', 'm_{2}', 'm_{3}'), ('m_{4}', 'm_{6}', 'c_{5}'), ('c_{1}', 'c_{2}', 'c_{13}'), ('m_{4}', 'm_{2}', 'm_{6}'), ('m_{4}', 'c_{3}', 'c_{4}'), ('c_{1}', 'c_{15}', 'c_{13}'), ('m_{6}', 'c_{5}', 'm_{3}'), ('c_{3}', 'm_{2}', 'm_{6}'), ('c_{3}', 'm_{6}', 'm_{3}'), ('m_{2}', 'm_{6}', 'm_{3}'), ('m_{4}', 'c_{3}', 'm_{5}'), ('m_{6}', 'c_{5}', 'm_{7}'), ('m_{2}', 'm_{6}', 'c_{5}'), ('m_{2}', 'c_{5}', 'm_{3}'), ('m_{4}', 'c_{5}', 'm_{3}'), ('c_{1}', 'c_{15}', 'c_{2}'), ('c_{3}', 'c_{5}', 'm_{3}')], 'c_{2}': [], 'c_{12}': [], 's_{10}': [], 'm_{7}': [], 'a_{4}': [], 'c_{14}': [], 'c_{18}': [], 'q_{2}': [('c_{8}', 'm_{10}', 'm_{12}'), ('m_{11}', 'c_{7}', 'm_{8}'), ('m_{9}', 'c_{9}', 'c_{10}'), ('m_{11}', 'c_{7}', 'c_{14}'), ('m_{11}', 'c_{8}', 'm_{10}'), ('c_{16}', 'm_{9}', 'c_{9}'), ('c_{7}', 'c_{8}', 'm_{10}'), ('m_{9}', 'c_{9}', 'm_{8}'), ('c_{8}', 'm_{10}', 'c_{10}'), ('m_{10}', 'm_{12}', 'c_{10}'), ('c_{11}', 'c_{16}', 'c_{12}'), ('m_{13}', 'm_{8}', 'c_{14}'), ('m_{11}', 'c_{7}', 'c_{8}'), ('m_{11}', 'c_{7}', 'm_{10}'), ('m_{11}', 'c_{7}', 'm_{13}'), ('c_{8}', 'm_{12}', 'c_{10}'), ('c_{16}', 'c_{12}', 'c_{14}'), ('c_{11}', 'c_{16}', 'c_{14}'), ('m_{11}', 'm_{13}', 'm_{8}'), ('c_{7}', 'm_{8}', 'c_{14}'), ('m_{9}', 'c_{10}', 'm_{8}'), ('c_{9}', 'c_{10}', 'm_{8}'), ('m_{11}', 'm_{8}', 'c_{14}'), ('c_{11}', 'c_{12}', 'c_{14}'), ('c_{7}', 'm_{13}', 'c_{14}'), ('c_{7}', 'm_{13}', 'm_{8}'), ('m_{11}', 'm_{13}', 'c_{14}')], 'm_{5}': [], 'v_{3}': [], 's_{9}': [], 's_{12}': []}
    #print "integrate(nabla Delta3 _rem) =", integrate_BR(BR_nabal_delta3_rem)

    print "\nderivative({a_{1}: [(m_{1},m_{1},m_{1})]}) =", derivative({'a_{1}': [('m_{1}','m_{1}','m_{1}')]}, BR_C)

    print "\nfacet_to_cells(v_{1}) =", facet_to_cells('v_{1}', BR_C)

    BR_last_rem = {'q_{2}': [('m_{11}', 'm_{12}', 'm_{8}'),
                             ('c_{11}', 'm_{12}', 'm_{8}'),
                             ('t_{3}', 'v_{5}', 'm_{8}'),
                             ('c_{7}', 'm_{12}', 'm_{8}'),
                             ('t_{3}', 'm_{12}', 'v_{10}'),
                             ('m_{10}', 'm_{12}', 'm_{8}'),
                             ('c_{8}', 'm_{12}', 'm_{8}'),
                             ('t_{3}', 'm_{12}', 'v_{9}'),
                             ('t_{3}', 'v_{8}', 'm_{8}')]}
    print "integrate(nabla Delta3 _last rem) =", integrate_BR(BR_last_rem)
    print "deriv(integrate(nabla Delta3 _last rem)) =", derivative(integrate_BR(BR_last_rem), BR_C)




if __name__ == '__main__':
    main()
