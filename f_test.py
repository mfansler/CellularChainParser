import numpy
import scipy.sparse as sp

from Coalgebra import Coalgebra

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


def compute_f(C, g):

    # create a map from cells to index
    basis = {el: i for (i, el) in enumerate([el for grp in C.groups.values() for el in grp])}

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

    inv_mat = rref_mat.tocsc()[:, n:].tocsr()

    def has_zombies(x):

        x_vec = [0]*n
        for el in x:
            x_vec[basis[el]] = 1

        zombies = inv_mat.dot(numpy.array(x_vec))[rank:]

        return zombies.any()

    # on f still needs to do a little more
    # has_zombies is just a helper function
    return has_zombies


# test data
C = Coalgebra(
    {0: ['v'], 1: ['a', 'b'], 2: ['aa', 'ab']},
    {'aa': {'b': 1}},
    {}
)

f = compute_f(C, {})

print "f(v) = ", f(['v'])
print "f(a) = ", f(['a'])
print "f(b) = ", f(['b'])
print "f(aa) = ", f(['aa'])
print "f(ab) = ", f(['ab'])
