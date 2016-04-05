import numpy
import scipy.sparse as sp

from collections import Counter
from itertools import combinations

from Coalgebra import Coalgebra

__author__ = 'mfansler'


def row_swap(A, r1, r2):

    tmp = A.getrow(r1).copy()
    A[r1] = A[r2]
    A[r2] = tmp


def mat_mod2(A):

    A.data[:] = numpy.fmod(A.data, 2)
    return A


def row_reduce_mod2(A, augment=0):

    if A.ndim != 2:
        raise Exception("require two dimensional matrix input, found ", A.ndim)

    A = A.tocsr()
    A = mat_mod2(A)
    rank = 0
    for i in range(A.shape[1] - augment):

        nzs = A.getcol(i).nonzero()[0]
        upper_nzs = [nz for nz in nzs if nz < rank]
        lower_nzs = [nz for nz in nzs if nz >= rank]

        if len(lower_nzs) > 0:

            row_swap(A, rank, lower_nzs[0])
            for nz in lower_nzs[1:]:
                A[nz, :] = mat_mod2(A[nz, :] + A[rank, :])

            if rank > 0:
                for nz in upper_nzs:
                    A[nz, :] = mat_mod2(A[nz, :] + A[rank, :])

            rank += 1

    return A, rank


def list_mod(ls, modulus=2):
    return [s for s, num in Counter(ls).items() if num % modulus]


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

        # convert to vector in established basis
        x_vec = [0]*n
        for el in x:
            x_vec[basis[el]] = 1

        # converts vector to Im[boundary] basis
        x_ker = inv_mat.dot(numpy.array(x_vec))

        if any(x_ker[rank:]):
            print "WARNING: Invalid integral!"
            print x, "contains non-vanishing component (cycle)"

        # returns boundary cells that contain x as boundary
        return [b for (b, v) in zip(ker_basis, x_ker[:rank]) if v % 2]

    return f, integrate1


def main():

    # test data toy
    DGC = Coalgebra(
        {0: ['v'], 1: ['a', 'b'], 2: ['aa', 'ab']},
        {'aa': {'b': 1}},
        {} # don't really care about coproduct definition

    )

    DGC_g = {'h1_0': ['a'], 'h0_0': ['v'], 'h2_0': ['ab']}

    f = generate_f_integral(DGC, DGC_g)

    print "DGC Toy"
    print "f(v) = ", f(['v'])
    print "f(a) = ", f(['a'])
    print "f(b) = ", f(['b'])
    print "f(aa) = ", f(['aa'])
    print "f(ab) = ", f(['ab'])
    print "f(a + b) = ", f(['a', 'b'])
    print "f(aa + ab) = ", f(['aa', 'ab'])


    # test data linked
    LNK = Coalgebra(
        {0: ['v'], 1: ['a', 'b'], 2: ['s', 't_{1}', 't_{2}'], 3: ['p', 'q']},
        {'q': {'s': 1, 't_{2}': 1, 't_{1}': 1}, 'p': {'s': 1}},
        {} # don't really care about coproduct definition
    )

    LNK_g = {'h1_0': ['a'], 'h0_0': ['v'], 'h2_0': ['t_{1}'], 'h1_1': ['b']}

    f = generate_f_integral(LNK, LNK_g)

    print "\n\nLINKED"
    print "f(v) = ", f(['v'])
    print "f(a) = ", f(['a'])
    print "f(b) = ", f(['b'])
    print "f(s) = ", f(['s'])
    print "f(t_{1}) = ", f(['t_{1}'])
    print "f(t_{2}) = ", f(['t_{2}'])
    print "f(p) = ", f(['p'])
    print "f(q) = ", f(['q'])

    BR_C = Coalgebra(
        {0: ['v_{1}', 'v_{2}', 'v_{3}', 'v_{4}', 'v_{5}', 'v_{6}', 'v_{7}', 'v_{8}', 'v_{9}', 'v_{10}', 'v_{11}'], 1: ['m_{1}', 'm_{2}', 'm_{3}', 'm_{4}', 'm_{5}', 'm_{6}', 'm_{7}', 'm_{8}', 'm_{9}', 'm_{10}', 'm_{11}', 'm_{12}', 'm_{13}', 'm_{14}', 'c_{1}', 'c_{2}', 'c_{3}', 'c_{4}', 'c_{5}', 'c_{6}', 'c_{7}', 'c_{8}', 'c_{9}', 'c_{10}', 'c_{11}', 'c_{12}', 'c_{13}', 'c_{14}', 'c_{15}', 'c_{16}', 'c_{17}', 'c_{18}'], 2: ['a_{1}', 'a_{2}', 'a_{3}', 'a_{4}', 'e_{1}', 'e_{2}', 's_{1}', 's_{2}', 's_{3}', 's_{4}', 's_{5}', 's_{6}', 's_{7}', 's_{8}', 's_{9}', 's_{10}', 's_{11}', 's_{12}', 't_{1}', 't_{2}', 't_{3}', 't_{4}', 't_{5}', 't_{6}', 't_{7}', 't_{8}'], 3: ['D', 'q_{1}', 'q_{2}', 'q_{3}', 'q_{4}']},
        {'m_{1}': {'v_{11}': 1, 'v_{1}': 1}, 'c_{5}': {'v_{7}': 1, 'v_{8}': 1}, 'm_{3}': {'v_{11}': 1, 'v_{8}': 1}, 'c_{13}': {'v_{11}': 1, 'v_{10}': 1}, 'c_{7}': {'v_{2}': 1, 'v_{3}': 1}, 'c_{3}': {'v_{2}': 1, 'v_{3}': 1}, 't_{6}': {'m_{7}': 1, 'c_{2}': 1, 'm_{6}': 1, 'c_{5}': 1, 'c_{6}': 1}, 't_{2}': {'c_{3}': 1, 'm_{5}': 1, 'c_{4}': 1, 'c_{1}': 1, 'm_{4}': 1}, 's_{4}': {'m_{9}': 1, 'c_{16}': 1, 'm_{12}': 1, 'c_{9}': 1}, 'c_{1}': {'v_{5}': 1, 'v_{1}': 1}, 'm_{9}': {'v_{7}': 1, 'v_{6}': 1}, 't_{4}': {'c_{7}': 1, 'c_{8}': 1, 'm_{5}': 1, 'c_{11}': 1, 'm_{4}': 1}, 's_{6}': {'c_{10}': 1, 'm_{7}': 1, 'c_{14}': 1, 'm_{3}': 1}, 'e_{2}': {'c_{18}': 1, 'c_{16}': 1, 'c_{14}': 1, 'c_{11}': 1, 'c_{12}': 1}, 'a_{3}': {'m_{1}': 1, 'c_{17}': 1}, 't_{7}': {'m_{9}': 1, 'c_{10}': 1, 'm_{8}': 1, 'c_{9}': 1, 'c_{12}': 1}, 'm_{12}': {'v_{5}': 1, 'v_{8}': 1}, 'c_{11}': {'v_{5}': 1, 'v_{1}': 1}, 'c_{15}': {'v_{5}': 1, 'v_{6}': 1}, 'a_{1}': {'c_{17}': 1, 'm_{14}': 1}, 'q_{3}': {'a_{3}': 1, 's_{1}': 1, 'e_{1}': 1, 's_{5}': 1, 't_{2}': 1, 's_{11}': 1, 't_{6}': 1}, 'c_{9}': {'v_{7}': 1, 'v_{8}': 1}, 't_{8}': {'c_{10}': 1, 'm_{7}': 1, 'm_{6}': 1, 'c_{9}': 1, 'c_{12}': 1}, 'c_{17}': {'v_{11}': 1, 'v_{1}': 1}, 's_{8}': {'c_{8}': 1, 'm_{13}': 1, 'm_{10}': 1, 'm_{12}': 1, 'c_{10}': 1}, 'c_{16}': {'v_{5}': 1, 'v_{6}': 1}, 'm_{10}': {'v_{4}': 1, 'v_{5}': 1}, 's_{11}': {'m_{2}': 1, 'c_{15}': 1, 'm_{5}': 1, 'c_{4}': 1}, 'm_{2}': {'v_{3}': 1, 'v_{6}': 1}, 'm_{14}': {'v_{11}': 1, 'v_{1}': 1}, 'm_{6}': {'v_{7}': 1, 'v_{6}': 1}, 'q_{2}': {'e_{2}': 1, 'a_{2}': 1, 's_{10}': 1, 's_{8}': 1, 's_{4}': 1, 't_{3}': 1, 't_{7}': 1}, 's_{3}': {'m_{9}': 1, 'c_{5}': 1, 'c_{15}': 1, 'm_{12}': 1}, 'q_{4}': {'e_{2}': 1, 's_{2}': 1, 't_{8}': 1, 'a_{4}': 1, 's_{6}': 1, 't_{4}': 1, 's_{12}': 1}, 'D': {'a_{3}': 1, 'a_{4}': 1, 'a_{1}': 1, 'a_{2}': 1}, 'c_{10}': {'v_{8}': 1, 'v_{9}': 1}, 't_{1}': {'m_{11}': 1, 'c_{3}': 1, 'm_{10}': 1, 'c_{4}': 1, 'c_{1}': 1}, 'c_{6}': {'v_{8}': 1, 'v_{9}': 1}, 't_{5}': {'m_{9}': 1, 'c_{5}': 1, 'm_{8}': 1, 'c_{2}': 1, 'c_{6}': 1}, 's_{5}': {'m_{7}': 1, 'c_{13}': 1, 'm_{3}': 1, 'c_{6}': 1}, 's_{2}': {'m_{1}': 1, 'm_{3}': 1, 'c_{7}': 1, 'c_{9}': 1, 'm_{6}': 1, 'm_{2}': 1, 'm_{4}': 1}, 't_{3}': {'m_{11}': 1, 'c_{8}': 1, 'm_{10}': 1, 'c_{7}': 1, 'c_{11}': 1}, 'm_{13}': {'v_{3}': 1, 'v_{9}': 1}, 'm_{8}': {'v_{10}': 1, 'v_{9}': 1}, 's_{7}': {'m_{13}': 1, 'm_{10}': 1, 'm_{12}': 1, 'c_{6}': 1, 'c_{4}': 1}, 'm_{11}': {'v_{2}': 1, 'v_{1}': 1}, 'c_{8}': {'v_{4}': 1, 'v_{3}': 1}, 'c_{4}': {'v_{4}': 1, 'v_{3}': 1}, 'a_{2}': {'c_{18}': 1, 'm_{14}': 1}, 's_{1}': {'m_{1}': 1, 'c_{5}': 1, 'm_{3}': 1, 'c_{3}': 1, 'm_{6}': 1, 'm_{2}': 1, 'm_{4}': 1}, 'e_{1}': {'c_{17}': 1, 'c_{15}': 1, 'c_{2}': 1, 'c_{13}': 1, 'c_{1}': 1}, 'q_{1}': {'s_{3}': 1, 'e_{1}': 1, 'a_{1}': 1, 't_{1}': 1, 't_{5}': 1, 's_{7}': 1, 's_{9}': 1}, 'c_{2}': {'v_{10}': 1, 'v_{6}': 1}, 'c_{12}': {'v_{10}': 1, 'v_{6}': 1}, 's_{10}': {'m_{11}': 1, 'c_{7}': 1, 'c_{14}': 1, 'm_{13}': 1, 'm_{8}': 1, 'm_{14}': 1}, 'm_{7}': {'v_{10}': 1, 'v_{9}': 1}, 'a_{4}': {'m_{1}': 1, 'c_{18}': 1}, 'c_{14}': {'v_{11}': 1, 'v_{10}': 1}, 'c_{18}': {'v_{11}': 1, 'v_{1}': 1}, 'm_{4}': {'v_{2}': 1, 'v_{1}': 1}, 'm_{5}': {'v_{4}': 1, 'v_{5}': 1}, 's_{9}': {'m_{11}': 1, 'c_{13}': 1, 'c_{3}': 1, 'm_{13}': 1, 'm_{8}': 1, 'm_{14}': 1}, 's_{12}': {'m_{2}': 1, 'c_{16}': 1, 'm_{5}': 1, 'c_{8}': 1}},
        {} # don't really care about coproduct definition
    )

    BR_g = {'h0_0': ['v_{1}'], 'h2_1': ['t_{5}', 't_{6}', 't_{7}', 't_{8}'], 'h2_0': ['t_{1}', 't_{2}', 't_{3}', 't_{4}'], 'h1_0': ['m_{11}', 'm_{4}'], 'h1_1': ['c_{3}', 'c_{7}'], 'h1_2': ['m_{6}', 'm_{9}']}

    f_BR = generate_f_integral(BR_C, BR_g)

    print "\n\nBorromean Rings"
    for c in [c for cells in BR_C.groups.values() for c in cells]:
        print "f(", c, ") = ", f_BR([c])

    for (l, r) in combinations(BR_C.groups[1], 2):
        print "f({} + {}) = {}".format(l, r, f_BR([l, r]))

    print "f(m_{4} + m_{11} + m_{8}) = ", f_BR(['m_{4}', 'm_{11}', 'm_{8}'])
    print "f(m_{4} + m_{11} + m_{14}) = ", f_BR(['m_{4}', 'm_{11}', 'm_{14}'])

if __name__ == '__main__':
    print "Name = ",  __name__
    main()
