import numpy
import scipy.sparse


def row_swap(A, r1, r2):
    #print "coo? ", scipy.sparse.isspmatrix_coo(A)
    #print r1
    #print r2
    tmp = A.getrow(r1).copy()
    A[r1] = A[r2]
    A[r2] = tmp

def mat_mod2(A):

    A.data[:] = numpy.fmod(A.data, 2)
    #print "mat_mod2"
    return A

def row_reduce_mod2(A, augment=-1):

    if A.ndim != 2:
        raise Exception("require two dimensional matrix input, found ", A.ndim)

    A = A.tocsr()
    A = mat_mod2(A)
    rank = 0
    for i in range(A.shape[1] + augment):

        nzs = A.getcol(i).nonzero()[0]
        upper_nzs = [nz for nz in nzs if nz < rank]
        lower_nzs = [nz for nz in nzs if nz >= rank]
        # print "rank = ", rank
        # print "upper_nzs = ", upper_nzs
        # print "lower_nzs = ", lower_nzs

        if len(lower_nzs) > 0:

            row_swap(A, rank, lower_nzs[0])
            # print "swapping: ", rank, lower_nzs[0]
            # print A.toarray()
            for nz in lower_nzs[1:]:
                A[nz, :] = mat_mod2(A[nz, :] + A[rank, :])
                #print "adding: ", nz, rank
                #print A.toarray()
            if rank > 0:
                for nz in upper_nzs:
                    A[nz, :] = mat_mod2(A[nz, :] + A[rank, :])
                    #print "adding: ", nz, rank
                    #print A.toarray()
            rank += 1

    return A

matrix1 = scipy.sparse.coo_matrix(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ])

print matrix1.toarray()

print row_reduce_mod2(matrix1).toarray()

# example how to build up above matrix incrementally

matrix2 = scipy.sparse.lil_matrix((12, 13), dtype=numpy.int8)

matrix2.rows[0] = [0]
matrix2.rows[1] = [1]
matrix2.rows[2] = [2]
matrix2.rows[3] = [2]
matrix2.rows[4] = [7]
matrix2.rows[5] = [8]
matrix2.rows[6] = [3]
matrix2.rows[7] = [4]
matrix2.rows[8] = [9]
matrix2.rows[9] = [10]
matrix2.rows[10] = [12]
matrix2.rows[11] = [0, 1, 2, 7, 8, 9]

matrix2.data = [[1]*len(row) for row in matrix2.rows]
print
print "M2 ="
print matrix2.transpose().toarray()

print
print "rref(M2) = "
print row_reduce_mod2(matrix2.transpose()).toarray()

