import numpy
import scipy.sparse as sp

from collections import Counter
from itertools import combinations, product
from copy import deepcopy
from random import shuffle
from multiprocessing import Pool

from Coalgebra import Coalgebra

__author__ = 'mfansler'


def expand_tuple_list(tp):
    def expand_tuple_helper(acc, tp_comp):
        if type(tp_comp) is list:
            return [x + (y,) for x in acc for y in tp_comp]
        return [x + (tp_comp,) for x in acc]

    if type(tp) is not tuple:
        raise TypeError("parameter must be a tuple")

    return reduce(expand_tuple_helper, tp, [tuple()])


def expand_map_all(xs):
    return {k: [tp for v in vs for tp in expand_tuple_list(v)] for k, vs in xs.iteritems() if vs}


def deep_freeze(x):
    type_x = type(x)
    if type_x is list or type_x is set:
        return frozenset([deep_freeze(el) for el in x])
    if type_x is tuple:
        return tuple(map(deep_freeze, x))

    return x


def deep_thaw(x):
    type_x = type(x)
    if type_x is list or type_x is set or type_x is frozenset:
        return [deep_thaw(el) for el in x]
    if type_x is tuple:
        return tuple(map(deep_thaw, x))

    return x


def unnest(ls):
    if type(ls) is list:
        if len(ls) == 1:
            return unnest(ls[0])
        return ls

    return [ls]


def tensor(*groups):

    maxes = [max(g.keys()) for g in groups]
    tensor_groups = {i: set() for i in range(sum(maxes) + 1)}

    for combin in product(*[range(m+1) for m in maxes]):
        tensor_groups[sum(combin)].update(
            product(*[group[combin[i]] for i, group in enumerate(groups)]))

    return tensor_groups


def factorize(tps, C):

    results = tps
    last_size = -1

    def is_cycle(chain):
        return not any(list_mod(derivative(chain, C)))

    def chains_equal(a, b):
        if type(a) is list and type(b) is list:
            return not any(list_mod(a + b))
        if type(a) == type(b):
            return a == b
        return False

    def tuple_diff_indices(tp1, tp2):
        return [i for i, (l, r) in enumerate(zip(tp1, tp2)) if not chains_equal(l, r)]

    def combine_component(a, b):
        if type(a) is list:
            if type(b) is list:
                return sorted(a + b)
            return sorted(a + [b])
        if type(b) is list:
            return sorted([a] + b)
        return sorted([a, b])

    def merge_adjacent_tuples(acc, tp):
        for i in range(len(acc)):
            idxs = tuple_diff_indices(acc[i], tp)
            if len(idxs) == 1 and is_cycle(list(acc[i][idxs[0]]) + list(tp[idxs[0]])):
                tp_list = list(acc[i])
                tp_list[idxs[0]] = combine_component(acc[i][idxs[0]], tp[idxs[0]])
                acc[i] = tuple(tp_list)
                return acc
        return acc + [tp]

    while len(results) - last_size:

        last_size = len(results)
        results = reduce(merge_adjacent_tuples, results, [])

    return results


def factorize_cycles(tps, C):

    if not tps:
        return tps

    results = tps

    # iterate through each tuple component
    for idx in range(len(tps[0])):

        # partition based on all other indices
        partition = {}
        for tp in results:
            key = deep_freeze(tp[:idx] + tp[idx+1:])
            if key in partition:
                partition[key].append(tp)
            else:
                partition[key] = [tp]

        results = []

        # iterate over the partitions, treating each as a queue
        for queue in partition.values():

            remaining = []

            for (d_tp, tp) in [(derivative(tp[idx], C), tp) for tp in queue]:
                if d_tp:
                    remaining.append((d_tp, tp))
                else:
                    # any unbounded elements can be placed directly in results
                    results.append(tp[:idx] + ([tp[idx]],) + tp[idx + 1:])

            cycle_len = 2
            queue = []
            while len(remaining) >= cycle_len:

                while remaining:
                    cur = remaining.pop(0)
                    cycle = None
                    for combs in combinations(range(len(remaining)), cycle_len -1):
                        # combine the derivatives
                        d_chain = cur[0] + [dx for i in combs for dx in remaining[i][0]]
                        # check if a simple chain is formed
                        if all([v == 2 for v in Counter(d_chain).values()]):
                            # get the different elements as a list
                            cycle_cmps = [cur[1][idx]] + [tp[1][idx] for j, tp in enumerate(remaining) if j in combs]
                            # merge them into a single cycle chain
                            cycle = cur[1][:idx] + (cycle_cmps, ) + cur[1][idx + 1:]
                            # append to results
                            results.append(cycle)
                            # remove what was merged from the list of remaining
                            remaining = [tp for j, tp in enumerate(remaining) if j not in combs]

                            # break out of the for loop
                            break

                    if cycle is None:
                        queue.append(cur)

                # increase cycle length
                cycle_len += 1

                # swap
                remaining, queue = queue, []

            # if any remainders, add them to the results
            results += [tp[:idx] + ([tp[idx]],) + tp[idx + 1:] for (_, tp) in remaining]

    return results


def group_boundary_chains(chains, C):

    num_cols = len(chains)
    num_rows = 0
    entries = []
    row_legend = {}

    # iterate over chains, creating a vector representation of their boundaries
    for chain in chains:
        entry = []
        d_chain = [t for dx in derivative(chain, C) for t in expand_tuple_list(dx)]
        d_chain = list_mod(d_chain)
        for dx in d_chain:
            frozen_dx = deep_freeze(dx)
            if frozen_dx not in row_legend:
                row_legend[frozen_dx] = num_rows
                num_rows += 1
            entry.append(row_legend[frozen_dx])
        entries.append(sorted(list_mod(entry)))

    # create matrix
    bd_mat = sp.lil_matrix((num_cols, num_rows), dtype=numpy.int8)
    bd_mat.rows = entries
    bd_mat.data = [[1]*len(row) for row in bd_mat.rows]
    bd_mat = bd_mat.transpose()

    # row reduce
    rref_mat, rank = row_reduce_mod2(bd_mat, augment=0)

    # use reduced matrix to group chains into boundary cycles
    ungrouped = range(num_cols)
    groups = []
    while ungrouped:
        group = []
        # pop columns off from right to left
        j = ungrouped.pop()
        group.append(chains[j])
        cur_col = rref_mat.getcol(j)
        for nz in cur_col.nonzero()[0]:
            if nz != j:
                left_col = rref_mat.getrow(nz).nonzero()[1][0]
                ungrouped.remove(left_col)
                group.append(chains[left_col])
        groups.append(group)

    return groups


def add_maps_mod_2(a, b):

    res = chain_map_mod(a)
    for k, vals in deepcopy(b).iteritems():
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
    r1_plus_r2 = []
    pos_r1 = 0
    pos_r2 = 0

    try:
        while True:
            if A.rows[r1][pos_r1] < A.rows[r2][pos_r2]:
                r1_plus_r2.append(A.rows[r1][pos_r1])
                pos_r1 += 1
            elif A.rows[r1][pos_r1] > A.rows[r2][pos_r2]:
                r1_plus_r2.append(A.rows[r2][pos_r2])
                pos_r2 += 1
            else:
                pos_r1 += 1
                pos_r2 += 1
    except IndexError:
        if pos_r1 == len(A.rows[r1]):
            r1_plus_r2.extend(A.rows[r2][pos_r2:])
        elif pos_r2 == len(A.rows[r2]):
            r1_plus_r2.extend(A.rows[r1][pos_r1:])

    A.rows[r2] = r1_plus_r2
    #A.rows[r2] = sorted(list_mod(A.rows[r1] + A.rows[r2]))
    A.data[r2] = [1]*len(A.rows[r2])


def parallel_row_sum((j, r1, r2)):
    r1_plus_r2 = []
    pos_r1 = 0
    pos_r2 = 0

    try:
        while True:
            if r1[pos_r1] < r2[pos_r2]:
                r1_plus_r2.append(r1[pos_r1])
                pos_r1 += 1
            elif r1[pos_r1] > r2[pos_r2]:
                r1_plus_r2.append(r2[pos_r2])
                pos_r2 += 1
            else:
                pos_r1 += 1
                pos_r2 += 1
    except IndexError:
        if pos_r1 == len(r1):
            r1_plus_r2.extend(r2[pos_r2:])
        elif pos_r2 == len(r2):
            r1_plus_r2.extend(r1[pos_r1:])

    return j, r1_plus_r2, [1]*len(r1_plus_r2)


def row_reduce_mod2(A, augment=0):

    if A.ndim != 2:
        raise Exception("require two dimensional matrix input, found ", A.ndim)

    # convert to lil matrix for efficient reduction
    A = A.tolil()

    # make sure everything is mod 2
    A.data = [[x % 2 for x in xs] for xs in A.data]

    # rank = the num of independent columns found so far
    rank = 0

    # this is used repeatedly, so makes sense to generate it only once
    row_range = range(A.shape[0])

    # multithreading worker pool for computing row sums
    p = Pool(8)

    for i in range(A.shape[1] - augment):

        # identify all rows with nonzero entries in i-th column
        # only those rows below the nonzero rows of the independent cols are independent
        nzs = [(j, len(A.rows[j])) for j in row_range if i in A.rows[j]]
        upper_nzs = [nz[0] for nz in nzs if nz[0] < rank]
        lower_nzs = [nz for nz in nzs if nz[0] >= rank]

        # progess information
        print "\rcolumn: {}/{}; num_lower_nonzero: {}; reducible: {}".format(i, A.shape[1], len(lower_nzs), i - rank),

        if len(lower_nzs) > 0:
            # the sparsest will be used
            sparsest = min(lower_nzs, key=lambda nz: nz[1])

            row_swap(A, rank, sparsest[0])
            sum_jobs = []
            for nz in [nz_pair[0] for nz_pair in lower_nzs if nz_pair != sparsest]:

                if nz != rank:
                    # if not from the rank position, then it hasn't moved
                    sum_jobs.append((nz, A.rows[rank], A.rows[nz]))
                elif sparsest[0] != nz:
                    # if it was in the rank row, and it was moved
                    # the main this is, we don't want to add the rank row to itself
                    sum_jobs.append((sparsest[0], A.rows[rank], A.rows[sparsest[0]]))

            if rank > 0:
                for nz in upper_nzs:
                    sum_jobs.append((nz, A.rows[rank], A.rows[nz]))

            # doesn't matter what order they are done in
            for j, row_j, data_j in p.imap_unordered(parallel_row_sum, sum_jobs, chunksize=100):
                A.rows[j] = row_j
                A.data[j] = data_j

            rank += 1

    # clean up the pool
    p.close()
    p.terminate()

    return A, rank


def ref_mod2(A, augment=0, eliminate=True):

    if A.ndim != 2:
        raise Exception("require two dimensional matrix input, found ", A.ndim)

    # convert to lil matrix for efficient reduction
    A = A.tolil()

    # make sure everything is mod 2
    A.data = [[x % 2 for x in xs] for xs in A.data]

    # rank = the num of independent columns found so far
    rank = 0

    # this is used repeatedly, so makes sense to generate it only once
    row_range = range(A.shape[0])

    # multithreading worker pool for computing row sums
    p = Pool(8)

    for i in range(A.shape[1] - augment):

        # only consider lower nonzeros, and only have to check first element in each row
        lower_nzs = [(j, len(A.rows[j])) for j in row_range[rank:] if A.rows[j] and A.rows[j][0] == i]

        # print some info on progress
        print "\rcolumn: {}/{}; num_lower_nonzero: {}; reducible: {}".format(i + 1, A.shape[1], len(lower_nzs), i - rank),

        if len(lower_nzs) > 0:
            # select the sparsest row to serve as representative
            # in order to minimize accumulation of nonzeros in matrix
            sparsest = min(lower_nzs, key=lambda nz: nz[1])
            row_swap(A, rank, sparsest[0])

            # work queue
            sum_jobs = []

            for nz in [nz_pair[0] for nz_pair in lower_nzs if nz_pair != sparsest]:

                if nz != rank:
                    # if not from the rank position, then it hasn't moved
                    sum_jobs.append((nz, A.rows[rank], A.rows[nz]))
                elif sparsest[0] != nz:
                    # if it was in the rank row, and it was moved
                    # the main this is, we don't want to add the rank row to itself
                    sum_jobs.append((sparsest[0], A.rows[rank], A.rows[sparsest[0]]))

            # carry out sums
            for j, row_j, data_j in p.imap_unordered(parallel_row_sum, sum_jobs, chunksize=100):
                A.rows[j] = row_j
                A.data[j] = data_j

            rank += 1

        elif eliminate:
            for j in range(rank):
                if i in A.rows[j]:
                    A.rows[j].remove(i)
                    A.data[j].pop()


    print "finished, now cleaning up"
    # clean up the pool
    p.close()
    p.terminate()

    return A, rank


def backsubstitute_mod2(ref_mat):
    # assumes that last column is the y column
    print "Back-substitution"
    # solution for (ref_mat[:,:-1]) * (x) = (ref_mat[:,-1])
    x = []

    # all nzs remaining
    nzs = ref_mat.getcol(-1).nonzero()[0]
    nzs = nzs.tolist()

    while len(nzs) > 0:
        print "\rlen(y) = {}; len(x) = {}".format(len(nzs), len(x)),

        row_idx = nzs[-1]
        col_idx = ref_mat.getrow(row_idx).nonzero()[1][0]
        x.append(col_idx)

        for i in ref_mat.getcol(col_idx).nonzero()[0]:
            if i in nzs:
                nzs.remove(i)
            else:
                nzs.append(i)
        nzs.sort()

    return x


def dot_mod2(A, x):
    # NOTE: assumes lil matrix

    y = []
    nzs = x.nonzero()[0]
    nzs = set(nzs)

    for i, r in enumerate(A.rows):

        if len(nzs.intersection(set(r))) % 2:
            y.append(i)

    # returns a list of all nonzero entries in product
    return y


def select_basis(A):
    cols = []
    rank = 0
    A = A.tocsc()
    row_red_op = sp.eye(A.shape[0])
    row_red_op = row_red_op.tolil()

    for i in range(A.shape[1]):
        #c = mat_mod2(row_red_op.dot(A.getcol(i)))

        #nzs = c.nonzero()[0]
        nzs = dot_mod2(row_red_op, A.getcol(i))
        upper_nzs = [j for j in nzs if j < rank]
        lower_nzs = [j for j in nzs if j >= rank]

        # print some info on progress
        print "\rcolumn: {}/{}; num_lower_nonzero: {}; reducible: {}".format(i, A.shape[1], len(lower_nzs), i - rank),

        if len(lower_nzs) > 0:
            cols.append(i)

            if rank != lower_nzs[0]:
                row_swap(row_red_op, rank, lower_nzs[0])

            lower_nzs.pop(0)

            for j in lower_nzs:
                add_rows(row_red_op, rank, j)

            for j in upper_nzs:
                add_rows(row_red_op, rank, j)

            rank += 1

    return cols, row_red_op, rank


def list_mod(ls, modulus=2):
    return [s for s, num in Counter(ls).iteritems() if num % modulus]


def chain_map_mod(xs, modulus=2):
    return {k: list_mod(ls, modulus=modulus) for k, ls in xs.iteritems()}


def facet_to_cells(facet, C):
    """
    Function to obtain a list of cells that a given cell is facet of.

    :param facet: cell from cell complex
    :param C: Coalgebra with cell complex that facet is in
    :return: a list of all cells in the complex for which the given cell is a facet
    """
    return {face for face, facets in C.differential.iteritems() if facet in facets and facets[facet] % 2}


def derivative(x, C):
    if type(x) is list:
        return [dy for y in x for dy in derivative(y, C)]
    if type(x) is tuple:
        return [tuple(x[:i]) + (derivative(x[i], C), ) + tuple(x[i + 1:]) for i in range(len(x))]
    if type(x) is dict:
        dx = {k: [] for k in x.keys()}

        # first handle raising cell-selecting map
        for k, vs in x.iteritems():
            for cell in facet_to_cells(k, C):
                if cell in dx:
                    dx[cell] += vs
                else:
                    dx[cell] = vs

            dx[k] += derivative(vs, C)
        return dx
    if x in C.differential:
        return [k for k, v in C.differential[x].iteritems() if v % 2]

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
    inv_basis = {v: k for k, v in basis.iteritems()}

    # store num cells
    n = len(basis)

    # prepare n x n incidence matrix
    # includes multiple dimensions, but it's sparse, so no efficiency lost
    inc_mat = sp.lil_matrix((n, n), dtype=numpy.int8)

    # enter incidences
    for el, bd in C.differential.iteritems():
        inc_mat.rows[basis[el]] = [basis[c] for c, i in bd.iteritems() if i % 2]
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
        bd = list_mod([dx for cell in x if cell in C.differential for dx in C.differential[cell].iteritems()], 2)
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
            expanded_xs = list_mod(expanded_xs)

        # otherwise, we need to factorize it, which improves performance
        else:
            expanded_xs = xs
            xs = factorize_cycles(list_mod(xs), C)
            xs = factorize(xs, C)

        # we now have a none empty list of tuples with list components
        shuffle(xs)
        for x in xs:
            # figure out which component in the first tuple can be integrated
            for i, x_cmp in enumerate(x):

                anti_x_cmp = integrate1(x_cmp)

                # if this component can't be integrated, continue the loop
                if anti_x_cmp is None:
                    continue

                # otherwise construct the anti_derivative that kills it
                else:
                    if i == 0:
                        anti_x = (anti_x_cmp,) + tuple(x[1:])
                    else:
                        anti_x = tuple(x[:i]) + (anti_x_cmp, ) + tuple(x[i + 1:])

                    anti_x = expand_tuple_list(anti_x)
                    # take the derivative of that anti-derivative and subtract from our list
                    anti_x_full_derivative = [dx_tp for dx in derivative(anti_x, C) for dx_tp in expand_tuple_list(dx) if all(dx)]
                    remainder = list_mod(anti_x_full_derivative + expanded_xs, 2)

                    # attempt to integrate that remaining portion on its own
                    #print len(remainder)
                    anti_rem = integrate(remainder) if len(remainder) < len(expanded_xs) else None

                    # if successful, then we have constructed a valid integral for xs
                    if anti_rem is not None:
                        # sweet
                        return anti_rem + anti_x

                    # otherwise loop back around and check for another component

            # that tuple could not be integrated on its own
            # loop back around and try the next one
        return None

    def integrate_chain_map(xs, allow_regress=True):
        # assuming map comes in factored
        expanded_map = chain_map_mod(expand_map_all(xs))
        best_distance = sum([len(vs) for vs in expanded_map.values()])
        print "\rbest_distance =", best_distance,
        if best_distance == 0:
            return {}

        frontier = []

        # generate initial frontier
        for cell, tps in expanded_map.iteritems():
            for tp in tps:

                # first generate anti-derivatives that kill chain components
                for i in range(len(tp)):
                    for anti_tp_cmp in facet_to_cells(tp[i], C):
                        anti_tp_map = {cell: [tp[:i] + (anti_tp_cmp,) + tp[i+1:]]}
                        der_anti_tp_map = chain_map_mod(expand_map_all(derivative(anti_tp_map, C)))
                        der_size = sum([len(vs) for vs in der_anti_tp_map.values()])
                        num_hits = sum([1 for k, vs in der_anti_tp_map.iteritems() for v in vs if k in expanded_map and v in expanded_map[k]])
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
                    num_hits = sum([1 for k, vs in der_anti_tp_map.iteritems() for v in vs if k in expanded_map and v in expanded_map[k]])
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


def matrix_reorder_rows_by_sparsity(A):

    ordering = sorted([(i, len(r), sum(r)) for i, r in enumerate(A.rows)], key=lambda (j, l, s): (l, s))

    B = sp.lil_matrix(A.shape, dtype=numpy.int8)
    for i in range(len(A.rows)):
        B.rows[i] = A.rows[ordering[i][0]]
        B.data[i] = A.data[ordering[i][0]]

    return B, {b_idx: a_idx for b_idx, (a_idx, _, _) in enumerate(ordering)}


def group_integrate(group, C):

    groups_inv = {cell: dim for dim, cells in C.groups.iteritems() for cell in cells}
    if not group:
        return []

    exp_group = [cell for chain in group for cell in expand_tuple_list(chain)]
    dim = len(group[0])

    dim_indices = set([tuple([groups_inv[cell] for cell in chain]) for chain in exp_group])
    anti_dim_indices = set([index[:i] + (index[i] + 1,) + index[i+1:] for index in dim_indices for i in range(dim)])

    sum_dimension = sum(next(iter(dim_indices)))

    tallies = [{} for _ in range(dim)]
    for chain in group:
        for i in range(dim):
            cur = chain[i]
            if type(cur) is list:
                for el in cur:
                    if groups_inv[el] in tallies[i]:
                        tallies[i][groups_inv[el]].append(el)
                    else:
                        tallies[i][groups_inv[el]] = [el]
            else:
                if groups_inv[cur] in tallies[i]:
                    tallies[i][groups_inv[cur]].append(cur)
                else:
                    tallies[i][groups_inv[cur]] = [cur]

    tallies = [{i: set(vs) for i, vs in components.iteritems()} for components in tallies]
    if any([len(tally) > 2 for tally in tallies]):

        anti_derivative_space = []
        for index in anti_dim_indices:
            anti_derivative_space += expand_tuple_list(tuple([C.groups[i] for i in index]))
        anti_derivative_space = list_mod(anti_derivative_space)

        num_cols = len(anti_derivative_space) + 1
        num_rows = 0
        entries = []
        row_legend = {}

        # iterate over possible components in anti-derivative space,
        # creating a vector representation of their derivatives
        for chain in anti_derivative_space:
            entry = []
            d_chain = [t for dx in derivative(chain, C) for t in expand_tuple_list(dx)]
            d_chain = list_mod(d_chain)
            for dx in d_chain:
                frozen_dx = deep_freeze(dx)
                if frozen_dx not in row_legend:
                    row_legend[frozen_dx] = num_rows
                    num_rows += 1
                entry.append(row_legend[frozen_dx])
            entries.append(sorted(list_mod(entry)))

        # include a temp dummy row
        entries.append(list(range(num_rows)))

        #entries.append(sorted([row_legend[deep_freeze(chain)] for chain in list_mod(exp_group)]))

        # create matrix
        bd_mat = sp.lil_matrix((num_cols, num_rows), dtype=numpy.int8)
        bd_mat.rows = entries
        bd_mat.data = [[1]*len(row) for row in bd_mat.rows]

        bd_mat, reordered_map = matrix_reorder_rows_by_sparsity(bd_mat)

        # replace the dummy row
        bd_mat.rows[-1] = sorted([row_legend[deep_freeze(chain)] for chain in list_mod(exp_group)])
        bd_mat.data[-1] = [1]*len(bd_mat.rows[-1])

        # transpose the matrix
        bd_mat = bd_mat.transpose()

        #cols, rr_op_mat, rank = select_basis(bd_mat)

        #sol = mat_mod2(rr_op_mat.dot(bd_mat.getcol(-1))).nonzero()[0]

        # row reduce
        #rref_mat, rank = row_reduce_mod2(bd_mat, augment=1)
        ref_mat, rank = ref_mod2(bd_mat, augment=1)
        print "reduced to row echelon form with rank", rank

        anti_derivative = backsubstitute_mod2(ref_mat)
        anti_derivative = [anti_derivative_space[reordered_map[col]] for col in anti_derivative]
        # solution = rref_mat.getcol(-1)
        # for nz in solution.nonzero()[0]:
        #     leftmost_col = rref_mat.getrow(nz).nonzero()[1][0]
        #     if leftmost_col == num_cols - 1:
        #         break
        #     anti_derivative.append(anti_derivative_space[leftmost_col])

        full_derivative = [dx_tp for dx in derivative(anti_derivative, C) for dx_tp in expand_tuple_list(dx) if all(dx)]
        full_derivative = list_mod(full_derivative)
        if not list_mod(full_derivative + exp_group):
            return anti_derivative
        else:
            print "\ngroup: ", group
            print "proposed antiderivative: ", factorize(factorize_cycles(anti_derivative, C), C)
            print "calculated derivative:", factorize(factorize_cycles(full_derivative, C), C)

        return None

    if any([len(tally) > 1 for tally in tallies]):
        # all possible anti-derivatives given the derivative chain
        dim_indices = expand_tuple_list(tuple([tally.keys() for tally in tallies]))
        dim_indices = [index for index in dim_indices if sum(index) == sum_dimension + 1]

        anti_derivative_space = []
        for index in dim_indices:
            anti_derivative_space.append(tuple([list(tallies[i][index[i]]) for i in range(dim)]))

        anti_derivative_space = [exp_tp for tp in anti_derivative_space for exp_tp in expand_tuple_list(tp)]
        anti_derivative_space = list_mod(anti_derivative_space)
        #anti_derivative_space = tuple([list(tally[max(tally.keys())]) for tally in tallies])
        #anti_derivative_space = expand_tuple_list(anti_derivative_space)

        num_cols = len(anti_derivative_space) + 1
        num_rows = 0
        entries = []
        row_legend = {}

        # iterate over possible components in anti-derivative space,
        # creating a vector representation of their derivatives
        for chain in anti_derivative_space:
            entry = []
            d_chain = [t for dx in derivative(chain, C) for t in expand_tuple_list(dx)]
            d_chain = list_mod(d_chain)
            for dx in d_chain:
                frozen_dx = deep_freeze(dx)
                if frozen_dx not in row_legend:
                    row_legend[frozen_dx] = num_rows
                    num_rows += 1
                entry.append(row_legend[frozen_dx])
            entries.append(sorted(list_mod(entry)))

        # append the group to the entries
        entries.append(sorted([row_legend[deep_freeze(chain)] for chain in list_mod(exp_group)]))

        # create matrix
        bd_mat = sp.lil_matrix((num_cols, num_rows), dtype=numpy.int8)
        bd_mat.rows = entries
        bd_mat.data = [[1]*len(row) for row in bd_mat.rows]
        bd_mat = bd_mat.transpose()

        # row reduce
        ref_mat, rank = ref_mod2(bd_mat, augment=1)

        anti_derivative = backsubstitute_mod2(ref_mat)
        anti_derivative = [anti_derivative_space[col] for col in anti_derivative]

        full_derivative = [dx_tp for dx in derivative(anti_derivative, C) for dx_tp in expand_tuple_list(dx) if all(dx)]
        full_derivative = list_mod(full_derivative)
        if not list_mod(full_derivative + exp_group):
            return anti_derivative
        else:
            print "\ngroup: ", group
            print "proposed antiderivative: ", factorize(factorize_cycles(anti_derivative, C), C)
            print "calculated derivative:", factorize(factorize_cycles(full_derivative, C), C)

    return None


def main():

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

    # test data for Delta_C3 computation
    # BR_nabla_delta_C3 = {'t_{2}': [('m_{4}', 'v_{2}', 'm_{5}'), ('v_{1}', 'c_{3}', 'c_{4}'), ('m_{4}', 'v_{4}', 'm_{5}'), ('c_{3}', 'c_{4}', 'v_{5}'), ('c_{3}', 'v_{4}', 'm_{5}'), ('c_{3}', 'v_{3}', 'm_{5}'), ('m_{4}', 'c_{3}', 'v_{5}'), ('v_{2}', 'c_{3}', 'c_{4}'), ('c_{3}', 'c_{4}', 'v_{4}'), ('v_{1}', 'c_{4}', 'm_{5}'), ('m_{4}', 'v_{3}', 'c_{4}'), ('m_{4}', 'c_{3}', 'v_{3}'), ('v_{1}', 'c_{3}', 'm_{5}'), ('m_{4}', 'c_{4}', 'v_{4}'), ('v_{3}', 'c_{4}', 'm_{5}'), ('v_{2}', 'c_{3}', 'm_{5}'), ('m_{4}', 'v_{2}', 'c_{4}'), ('m_{4}', 'c_{4}', 'v_{5}')], 's_{4}': [('c_{16}', 'v_{6}', 'c_{9}'), ('v_{6}', 'm_{9}', 'c_{9}'), ('v_{5}', 'm_{9}', 'c_{9}'), ('c_{16}', 'm_{9}', 'v_{8}'), ('c_{16}', 'm_{9}', 'v_{7}'), ('c_{16}', 'v_{7}', 'c_{9}')], 't_{6}': [('c_{5}', 'v_{9}', 'm_{7}'), ('m_{6}', 'c_{5}', 'v_{10}'), ('v_{8}', 'c_{6}', 'm_{7}'), ('m_{6}', 'v_{7}', 'm_{7}'), ('m_{6}', 'v_{8}', 'c_{6}'), ('c_{5}', 'v_{8}', 'm_{7}'), ('m_{6}', 'v_{7}', 'c_{6}'), ('c_{5}', 'c_{6}', 'v_{9}'), ('v_{6}', 'c_{5}', 'm_{7}'), ('v_{6}', 'c_{5}', 'c_{6}'), ('v_{7}', 'c_{5}', 'c_{6}'), ('v_{6}', 'c_{6}', 'm_{7}'), ('m_{6}', 'v_{9}', 'm_{7}'), ('v_{7}', 'c_{5}', 'm_{7}'), ('m_{6}', 'c_{6}', 'v_{10}'), ('m_{6}', 'c_{6}', 'v_{9}'), ('c_{5}', 'c_{6}', 'v_{10}'), ('m_{6}', 'c_{5}', 'v_{8}')], 't_{4}': [('m_{4}', 'v_{2}', 'm_{5}'), ('m_{4}', 'v_{4}', 'm_{5}'), ('c_{7}', 'c_{8}', 'v_{5}'), ('m_{4}', 'c_{7}', 'v_{3}'), ('c_{7}', 'v_{3}', 'm_{5}'), ('m_{4}', 'c_{7}', 'v_{5}'), ('m_{4}', 'c_{8}', 'v_{5}'), ('c_{7}', 'v_{4}', 'm_{5}'), ('m_{4}', 'v_{3}', 'c_{8}'), ('c_{7}', 'c_{8}', 'v_{4}'), ('v_{1}', 'c_{8}', 'm_{5}'), ('m_{4}', 'v_{2}', 'c_{8}'), ('v_{2}', 'c_{7}', 'm_{5}'), ('v_{1}', 'c_{7}', 'm_{5}'), ('v_{3}', 'c_{8}', 'm_{5}'), ('v_{1}', 'c_{7}', 'c_{8}'), ('v_{2}', 'c_{7}', 'c_{8}'), ('m_{4}', 'c_{8}', 'v_{4}')], 's_{6}': [('c_{10}', 'm_{7}', 'v_{11}'), ('c_{10}', 'm_{7}', 'v_{10}'), ('c_{10}', 'v_{10}', 'c_{14}'), ('v_{8}', 'm_{7}', 'c_{14}'), ('v_{9}', 'm_{7}', 'c_{14}'), ('c_{10}', 'v_{9}', 'c_{14}')], 'e_{2}': [('c_{11}', 'c_{12}', 'v_{10}'), ('c_{11}', 'c_{12}', 'v_{11}'), ('c_{16}', 'c_{12}', 'v_{11}'), ('c_{11}', 'v_{6}', 'c_{12}'), ('c_{11}', 'c_{16}', 'v_{11}'), ('v_{5}', 'c_{16}', 'c_{12}'), ('v_{1}', 'c_{16}', 'c_{14}'), ('c_{16}', 'v_{6}', 'c_{14}'), ('c_{11}', 'c_{16}', 'v_{6}'), ('v_{1}', 'c_{16}', 'c_{12}'), ('v_{6}', 'c_{12}', 'c_{14}'), ('c_{11}', 'v_{10}', 'c_{14}'), ('c_{16}', 'v_{10}', 'c_{14}'), ('c_{11}', 'v_{5}', 'c_{14}'), ('v_{5}', 'c_{16}', 'c_{14}'), ('c_{11}', 'v_{5}', 'c_{12}'), ('v_{1}', 'c_{12}', 'c_{14}'), ('c_{16}', 'c_{12}', 'v_{10}')], 's_{2}': [('c_{7}', 'v_{7}', 'c_{9}'), ('c_{7}', 'm_{6}', 'v_{7}'), ('v_{3}', 'm_{2}', 'm_{6}'), ('v_{1}', 'c_{7}', 'm_{6}'), ('v_{1}', 'm_{6}', 'm_{3}'), ('m_{2}', 'v_{7}', 'c_{9}'), ('m_{4}', 'c_{9}', 'v_{11}'), ('m_{2}', 'v_{6}', 'c_{9}'), ('v_{2}', 'c_{7}', 'c_{9}'), ('m_{4}', 'v_{3}', 'm_{2}'), ('m_{4}', 'm_{2}', 'v_{11}'), ('v_{1}', 'c_{7}', 'm_{3}'), ('v_{1}', 'c_{7}', 'm_{2}'), ('m_{4}', 'v_{2}', 'c_{9}'), ('v_{1}', 'm_{2}', 'm_{3}'), ('v_{2}', 'c_{7}', 'm_{6}'), ('m_{2}', 'v_{6}', 'm_{3}'), ('v_{6}', 'm_{6}', 'c_{9}'), ('m_{4}', 'v_{6}', 'm_{6}'), ('c_{7}', 'v_{3}', 'm_{3}'), ('m_{4}', 'v_{7}', 'c_{9}'), ('m_{2}', 'm_{6}', 'v_{7}'), ('v_{7}', 'c_{9}', 'm_{3}'), ('v_{1}', 'c_{7}', 'c_{9}'), ('c_{7}', 'c_{9}', 'v_{11}'), ('m_{6}', 'v_{8}', 'm_{3}'), ('m_{6}', 'v_{7}', 'm_{3}'), ('m_{4}', 'v_{2}', 'm_{6}'), ('m_{4}', 'c_{9}', 'v_{8}'), ('v_{1}', 'm_{2}', 'c_{9}'), ('c_{7}', 'v_{6}', 'm_{6}'), ('v_{3}', 'm_{2}', 'c_{9}'), ('m_{4}', 'v_{2}', 'm_{2}'), ('c_{7}', 'v_{3}', 'c_{9}'), ('m_{4}', 'm_{2}', 'v_{6}'), ('v_{1}', 'm_{6}', 'c_{9}'), ('c_{7}', 'm_{2}', 'v_{6}'), ('c_{7}', 'c_{9}', 'v_{8}'), ('m_{2}', 'c_{9}', 'v_{11}'), ('m_{6}', 'c_{9}', 'v_{11}'), ('v_{2}', 'c_{7}', 'm_{2}'), ('m_{4}', 'm_{6}', 'v_{7}'), ('c_{7}', 'm_{2}', 'v_{11}'), ('m_{4}', 'c_{7}', 'v_{11}'), ('m_{4}', 'v_{2}', 'm_{3}'), ('m_{6}', 'c_{9}', 'v_{8}'), ('m_{4}', 'c_{7}', 'v_{3}'), ('c_{7}', 'v_{8}', 'm_{3}'), ('m_{2}', 'c_{9}', 'v_{8}'), ('m_{4}', 'm_{6}', 'v_{11}'), ('v_{3}', 'm_{2}', 'm_{3}'), ('v_{2}', 'c_{7}', 'm_{3}'), ('v_{6}', 'm_{6}', 'm_{3}'), ('v_{1}', 'c_{9}', 'm_{3}'), ('c_{7}', 'm_{6}', 'v_{11}'), ('c_{7}', 'v_{3}', 'm_{6}'), ('m_{4}', 'v_{8}', 'm_{3}'), ('m_{2}', 'v_{8}', 'm_{3}'), ('v_{1}', 'm_{2}', 'm_{6}'), ('m_{2}', 'm_{6}', 'v_{11}')], 'q_{3}': [('c_{4}', 'm_{5}', 'c_{15}'), ('s_{11}', 'v_{6}', 'm_{3}'), ('c_{3}', 's_{11}', 'v_{11}'), ('c_{15}', 'c_{5}', 'm_{7}'), ('s_{11}', 'c_{5}', 'v_{8}'), ('v_{3}', 's_{11}', 'm_{3}'), ('m_{6}', 'c_{6}', 'm_{7}'), ('c_{3}', 's_{11}', 'v_{6}'), ('v_{1}', 'c_{3}', 's_{11}'), ('v_{1}', 's_{11}', 'c_{5}'), ('c_{5}', 'c_{6}', 'm_{7}'), ('c_{15}', 'c_{5}', 'c_{6}'), ('v_{1}', 's_{11}', 'm_{3}'), ('s_{11}', 'v_{7}', 'c_{5}'), ('c_{3}', 'm_{5}', 'm_{6}'), ('c_{1}', 'v_{6}', 't_{6}'), ('c_{1}', 'v_{8}', 's_{5}'), ('v_{1}', 'c_{15}', 't_{6}'), ('c_{1}', 'm_{6}', 'c_{5}'), ('c_{15}', 'm_{7}', 'c_{13}'), ('m_{4}', 'c_{3}', 'm_{6}'), ('v_{1}', 's_{11}', 'm_{6}'), ('c_{3}', 'c_{4}', 'm_{5}'), ('v_{2}', 'c_{3}', 's_{11}'), ('t_{2}', 'v_{5}', 'm_{6}'), ('m_{4}', 'c_{4}', 'm_{3}'), ('c_{1}', 't_{6}', 'v_{11}'), ('m_{6}', 'v_{7}', 's_{5}'), ('m_{4}', 'm_{5}', 'c_{5}'), ('c_{15}', 'v_{8}', 's_{5}'), ('v_{1}', 't_{6}', 'c_{13}'), ('t_{2}', 'c_{5}', 'v_{8}'), ('m_{4}', 'm_{5}', 'm_{3}'), ('t_{2}', 'c_{15}', 'v_{11}'), ('m_{4}', 'c_{3}', 'm_{3}'), ('m_{5}', 'c_{15}', 'm_{6}'), ('m_{5}', 'c_{15}', 'm_{3}'), ('t_{2}', 'm_{6}', 'v_{7}'), ('c_{15}', 'm_{6}', 'c_{6}'), ('c_{4}', 'c_{15}', 'm_{6}'), ('m_{4}', 's_{11}', 'v_{11}'), ('m_{6}', 'v_{8}', 's_{5}'), ('m_{4}', 'c_{3}', 'c_{5}'), ('m_{5}', 'c_{15}', 'c_{5}'), ('v_{1}', 'c_{5}', 's_{5}'), ('c_{3}', 'c_{4}', 'm_{6}'), ('m_{4}', 'c_{4}', 'm_{5}'), ('c_{15}', 'c_{6}', 'c_{13}'), ('c_{1}', 'm_{6}', 'm_{7}'), ('c_{3}', 'c_{4}', 'm_{3}'), ('s_{11}', 'm_{6}', 'v_{11}'), ('t_{2}', 'c_{5}', 'v_{11}'), ('c_{3}', 'c_{4}', 'c_{5}'), ('c_{1}', 'v_{5}', 's_{5}'), ('c_{15}', 'm_{6}', 'c_{5}'), ('t_{2}', 'c_{15}', 'v_{6}'), ('c_{6}', 'm_{7}', 'c_{13}'), ('m_{4}', 'v_{3}', 's_{11}'), ('v_{6}', 'm_{6}', 's_{5}'), ('c_{1}', 'm_{7}', 'c_{13}'), ('c_{3}', 'm_{5}', 'c_{5}'), ('v_{6}', 't_{6}', 'c_{13}'), ('m_{4}', 's_{11}', 'v_{6}'), ('m_{4}', 'c_{4}', 'm_{6}'), ('t_{2}', 'v_{5}', 'm_{3}'), ('c_{1}', 'c_{5}', 'c_{6}'), ('m_{6}', 'c_{5}', 'c_{13}'), ('c_{3}', 'm_{5}', 'm_{3}'), ('s_{11}', 'c_{5}', 'v_{11}'), ('m_{4}', 'm_{5}', 'm_{6}'), ('c_{15}', 'm_{6}', 'm_{7}'), ('t_{2}', 'v_{5}', 'c_{5}'), ('c_{1}', 'c_{6}', 'c_{13}'), ('v_{5}', 'c_{15}', 's_{5}'), ('v_{5}', 'c_{15}', 't_{6}'), ('c_{1}', 'v_{5}', 't_{6}'), ('t_{2}', 'v_{7}', 'c_{5}'), ('s_{11}', 'v_{8}', 'm_{3}'), ('c_{4}', 'c_{15}', 'm_{3}'), ('m_{4}', 'c_{3}', 'c_{15}'), ('c_{15}', 't_{6}', 'v_{11}'), ('t_{2}', 'v_{8}', 'm_{3}'), ('m_{4}', 'c_{4}', 'c_{5}'), ('c_{4}', 'c_{15}', 'c_{5}'), ('v_{1}', 'c_{15}', 's_{5}'), ('c_{15}', 'v_{6}', 's_{5}'), ('t_{2}', 'm_{6}', 'v_{11}'), ('m_{4}', 'v_{2}', 's_{11}'), ('v_{7}', 'c_{5}', 's_{5}'), ('v_{1}', 'm_{6}', 's_{5}'), ('c_{15}', 't_{6}', 'v_{10}'), ('t_{2}', 'v_{6}', 'm_{6}'), ('s_{11}', 'v_{6}', 'c_{5}'), ('c_{1}', 't_{6}', 'v_{10}'), ('c_{1}', 'c_{5}', 'm_{7}'), ('v_{3}', 's_{11}', 'c_{5}'), ('c_{1}', 'm_{6}', 'c_{6}'), ('s_{11}', 'm_{6}', 'v_{7}'), ('v_{3}', 's_{11}', 'm_{6}')], 's_{8}': [('v_{4}', 'm_{10}', 'm_{12}'), ('v_{3}', 'm_{10}', 'm_{12}'), ('m_{10}', 'm_{12}', 'v_{9}'), ('v_{4}', 'm_{10}', 'c_{10}'), ('v_{3}', 'm_{10}', 'c_{10}'), ('c_{8}', 'm_{12}', 'v_{8}'), ('m_{10}', 'v_{5}', 'c_{10}'), ('m_{10}', 'v_{8}', 'c_{10}'), ('c_{8}', 'm_{12}', 'v_{9}'), ('v_{5}', 'm_{12}', 'c_{10}'), ('c_{8}', 'v_{8}', 'c_{10}'), ('c_{8}', 'm_{10}', 'v_{5}'), ('c_{8}', 'v_{4}', 'c_{10}'), ('c_{8}', 'v_{5}', 'm_{12}'), ('v_{3}', 'm_{12}', 'c_{10}'), ('m_{10}', 'm_{12}', 'v_{8}'), ('c_{8}', 'v_{4}', 'm_{12}'), ('c_{8}', 'm_{10}', 'v_{9}')], 'q_{2}': [('m_{11}', 'v_{2}', 's_{8}'), ('m_{11}', 's_{8}', 'v_{9}'), ('s_{4}', 'c_{10}', 'v_{9}'), ('s_{8}', 'v_{9}', 'c_{14}'), ('m_{12}', 'c_{10}', 'c_{14}'), ('t_{3}', 'm_{12}', 'v_{8}'), ('c_{8}', 'm_{10}', 'm_{12}'), ('c_{7}', 'm_{12}', 'c_{10}'), ('t_{3}', 'm_{8}', 'v_{10}'), ('m_{10}', 'm_{12}', 'm_{8}'), ('c_{8}', 'c_{10}', 'm_{8}'), ('c_{8}', 'm_{12}', 'c_{14}'), ('c_{11}', 'c_{16}', 'm_{9}'), ('s_{4}', 'm_{8}', 'v_{11}'), ('c_{11}', 'm_{9}', 'm_{8}'), ('c_{11}', 's_{4}', 'v_{11}'), ('t_{3}', 'm_{12}', 'v_{11}'), ('m_{11}', 'c_{8}', 'm_{8}'), ('c_{11}', 'c_{10}', 'm_{8}'), ('c_{16}', 't_{7}', 'v_{10}'), ('v_{3}', 's_{8}', 'm_{8}'), ('c_{16}', 't_{7}', 'v_{11}'), ('v_{1}', 'c_{16}', 't_{7}'), ('t_{3}', 'm_{8}', 'v_{11}'), ('v_{1}', 's_{4}', 'm_{8}'), ('t_{3}', 'v_{8}', 'c_{10}'), ('c_{7}', 'm_{10}', 'c_{14}'), ('c_{7}', 'c_{8}', 'm_{8}'), ('v_{5}', 's_{4}', 'c_{14}'), ('m_{10}', 'c_{10}', 'c_{14}'), ('m_{11}', 'v_{3}', 's_{8}'), ('t_{3}', 'v_{9}', 'm_{8}'), ('t_{3}', 'v_{5}', 'c_{14}'), ('t_{3}', 'v_{5}', 'm_{8}'), ('c_{10}', 'm_{8}', 'c_{14}'), ('v_{5}', 'c_{16}', 't_{7}'), ('m_{11}', 'c_{7}', 'm_{8}'), ('t_{3}', 'v_{10}', 'c_{14}'), ('m_{10}', 'm_{12}', 'c_{14}'), ('m_{10}', 'c_{10}', 'm_{8}'), ('c_{11}', 'v_{5}', 't_{7}'), ('v_{1}', 't_{7}', 'c_{14}'), ('m_{9}', 'c_{9}', 'c_{10}'), ('c_{11}', 'v_{6}', 't_{7}'), ('s_{4}', 'v_{8}', 'c_{14}'), ('c_{16}', 'm_{9}', 'c_{14}'), ('c_{16}', 'c_{9}', 'c_{14}'), ('v_{1}', 's_{4}', 'c_{14}'), ('m_{12}', 'c_{10}', 'm_{8}'), ('s_{4}', 'm_{8}', 'v_{10}'), ('v_{5}', 's_{4}', 'c_{10}'), ('m_{11}', 'c_{8}', 'c_{14}'), ('s_{4}', 'c_{10}', 'v_{11}'), ('s_{4}', 'v_{8}', 'm_{8}'), ('m_{11}', 'c_{7}', 'c_{14}'), ('m_{11}', 's_{8}', 'v_{11}'), ('c_{8}', 'c_{10}', 'c_{14}'), ('m_{9}', 'm_{8}', 'c_{14}'), ('c_{7}', 's_{8}', 'v_{11}'), ('t_{3}', 'c_{10}', 'v_{11}'), ('c_{9}', 'm_{8}', 'c_{14}'), ('m_{11}', 'c_{8}', 'm_{10}'), ('c_{16}', 'm_{9}', 'c_{9}'), ('v_{6}', 't_{7}', 'c_{14}'), ('t_{3}', 'v_{5}', 'c_{10}'), ('s_{8}', 'm_{8}', 'v_{11}'), ('m_{11}', 'c_{7}', 'm_{12}'), ('c_{11}', 't_{7}', 'v_{11}'), ('c_{7}', 'm_{10}', 'm_{8}'), ('s_{8}', 'm_{8}', 'v_{10}'), ('c_{7}', 'c_{8}', 'm_{10}'), ('v_{1}', 's_{4}', 'c_{10}'), ('m_{11}', 'm_{12}', 'c_{10}'), ('m_{11}', 'c_{7}', 'c_{10}'), ('s_{8}', 'v_{10}', 'c_{14}'), ('v_{1}', 's_{8}', 'c_{14}'), ('m_{11}', 'm_{10}', 'c_{14}'), ('c_{8}', 'm_{12}', 'm_{8}'), ('v_{2}', 'c_{7}', 's_{8}'), ('c_{11}', 's_{4}', 'v_{8}'), ('m_{9}', 'c_{9}', 'm_{8}'), ('s_{4}', 'v_{9}', 'm_{8}'), ('v_{1}', 's_{8}', 'm_{8}'), ('m_{11}', 'm_{10}', 'm_{8}'), ('c_{11}', 't_{7}', 'v_{10}'), ('v_{5}', 's_{4}', 'm_{8}'), ('c_{11}', 'c_{9}', 'c_{10}'), ('c_{9}', 'c_{10}', 'c_{14}'), ('c_{11}', 'c_{16}', 'c_{9}'), ('m_{9}', 'c_{10}', 'c_{14}'), ('c_{11}', 'c_{9}', 'm_{8}'), ('t_{3}', 'c_{10}', 'v_{9}'), ('c_{7}', 'c_{8}', 'c_{14}'), ('c_{11}', 'm_{9}', 'c_{10}'), ('s_{4}', 'v_{10}', 'c_{14}'), ('v_{1}', 'c_{7}', 's_{8}'), ('c_{8}', 'm_{10}', 'c_{10}'), ('c_{7}', 's_{8}', 'v_{9}'), ('v_{3}', 's_{8}', 'c_{14}'), ('c_{16}', 'c_{10}', 'm_{8}')], 's_{11}': [('c_{4}', 'm_{5}', 'v_{5}'), ('v_{3}', 'm_{5}', 'c_{15}'), ('c_{4}', 'v_{4}', 'c_{15}'), ('v_{4}', 'm_{5}', 'c_{15}'), ('c_{4}', 'v_{5}', 'c_{15}'), ('c_{4}', 'm_{5}', 'v_{6}')], 'q_{4}': [('c_{11}', 'm_{7}', 'c_{14}'), ('t_{4}', 'c_{16}', 'v_{6}'), ('c_{11}', 'c_{10}', 'c_{14}'), ('c_{11}', 'v_{5}', 't_{8}'), ('m_{4}', 'c_{7}', 'm_{6}'), ('c_{7}', 'c_{8}', 'm_{6}'), ('c_{7}', 'm_{5}', 'm_{3}'), ('m_{4}', 'c_{8}', 'm_{3}'), ('c_{7}', 'c_{8}', 'm_{3}'), ('t_{4}', 'v_{8}', 'm_{3}'), ('c_{11}', 't_{8}', 'v_{11}'), ('c_{10}', 'm_{7}', 'c_{14}'), ('c_{7}', 's_{12}', 'v_{11}'), ('m_{4}', 'm_{5}', 'c_{9}'), ('m_{6}', 'c_{9}', 'c_{14}'), ('m_{4}', 's_{12}', 'v_{6}'), ('t_{4}', 'v_{5}', 'm_{3}'), ('c_{7}', 'm_{5}', 'c_{9}'), ('c_{16}', 'v_{8}', 's_{6}'), ('c_{11}', 'v_{8}', 's_{6}'), ('m_{4}', 'c_{8}', 'm_{5}'), ('v_{1}', 's_{12}', 'm_{6}'), ('c_{11}', 'v_{6}', 't_{8}'), ('m_{6}', 'v_{8}', 's_{6}'), ('v_{5}', 'c_{16}', 't_{8}'), ('t_{4}', 'm_{6}', 'v_{11}'), ('v_{3}', 's_{12}', 'c_{9}'), ('v_{1}', 'c_{7}', 's_{12}'), ('m_{4}', 'c_{8}', 'c_{9}'), ('m_{4}', 'm_{5}', 'm_{6}'), ('c_{9}', 'c_{10}', 'm_{7}'), ('t_{4}', 'm_{6}', 'v_{7}'), ('v_{6}', 't_{8}', 'c_{14}'), ('v_{3}', 's_{12}', 'm_{3}'), ('t_{4}', 'v_{6}', 'm_{6}'), ('m_{4}', 'v_{3}', 's_{12}'), ('t_{4}', 'c_{9}', 'v_{11}'), ('v_{1}', 'c_{9}', 's_{6}'), ('c_{16}', 't_{8}', 'v_{10}'), ('v_{1}', 't_{8}', 'c_{14}'), ('m_{6}', 'c_{10}', 'm_{7}'), ('c_{16}', 'c_{10}', 'c_{14}'), ('c_{7}', 'c_{8}', 'm_{5}'), ('m_{4}', 'c_{7}', 'c_{16}'), ('m_{5}', 'c_{16}', 'c_{9}'), ('m_{6}', 'v_{7}', 's_{6}'), ('c_{11}', 't_{8}', 'v_{10}'), ('s_{12}', 'c_{9}', 'v_{11}'), ('t_{4}', 'v_{7}', 'c_{9}'), ('v_{1}', 'c_{16}', 't_{8}'), ('s_{12}', 'c_{9}', 'v_{8}'), ('t_{4}', 'c_{9}', 'v_{8}'), ('s_{12}', 'v_{6}', 'c_{9}'), ('s_{12}', 'v_{8}', 'm_{3}'), ('v_{6}', 'm_{6}', 's_{6}'), ('c_{11}', 'm_{6}', 'c_{9}'), ('s_{12}', 'v_{6}', 'm_{3}'), ('m_{5}', 'c_{16}', 'm_{3}'), ('m_{5}', 'c_{16}', 'm_{6}'), ('c_{16}', 'v_{6}', 's_{6}'), ('c_{16}', 'm_{6}', 'c_{10}'), ('c_{8}', 'm_{5}', 'c_{16}'), ('c_{11}', 'c_{9}', 'm_{7}'), ('c_{11}', 'm_{6}', 'c_{10}'), ('m_{4}', 'v_{2}', 's_{12}'), ('v_{7}', 'c_{9}', 's_{6}'), ('c_{16}', 'm_{6}', 'c_{9}'), ('c_{16}', 'm_{6}', 'm_{7}'), ('c_{16}', 'c_{9}', 'c_{10}'), ('m_{4}', 'c_{8}', 'm_{6}'), ('s_{12}', 'v_{7}', 'c_{9}'), ('t_{4}', 'v_{5}', 'm_{6}'), ('c_{16}', 't_{8}', 'v_{11}'), ('v_{1}', 'c_{16}', 's_{6}'), ('m_{4}', 's_{12}', 'v_{11}'), ('c_{11}', 'v_{5}', 's_{6}'), ('s_{12}', 'm_{6}', 'v_{7}'), ('c_{8}', 'c_{16}', 'm_{3}'), ('s_{12}', 'm_{6}', 'v_{11}'), ('c_{11}', 'm_{6}', 'm_{7}'), ('c_{7}', 'm_{5}', 'm_{6}'), ('v_{1}', 'm_{6}', 's_{6}'), ('v_{1}', 's_{12}', 'm_{3}'), ('v_{3}', 's_{12}', 'm_{6}'), ('v_{2}', 'c_{7}', 's_{12}'), ('c_{11}', 'c_{9}', 'c_{10}'), ('c_{16}', 'c_{9}', 'm_{7}'), ('v_{1}', 's_{12}', 'c_{9}'), ('c_{8}', 'c_{16}', 'm_{6}'), ('m_{4}', 'm_{5}', 'm_{3}'), ('c_{7}', 's_{12}', 'v_{6}'), ('t_{4}', 'c_{16}', 'v_{11}'), ('m_{4}', 'c_{7}', 'c_{9}'), ('c_{7}', 'c_{8}', 'c_{9}'), ('v_{5}', 'c_{16}', 's_{6}'), ('c_{8}', 'c_{16}', 'c_{9}'), ('c_{16}', 'm_{7}', 'c_{14}'), ('m_{4}', 'c_{7}', 'm_{3}'), ('t_{4}', 'v_{5}', 'c_{9}')], 't_{1}': [('m_{11}', 'v_{2}', 'm_{10}'), ('v_{1}', 'c_{3}', 'c_{4}'), ('c_{3}', 'c_{4}', 'v_{5}'), ('m_{11}', 'c_{3}', 'v_{5}'), ('m_{11}', 'v_{4}', 'm_{10}'), ('m_{11}', 'c_{3}', 'v_{3}'), ('m_{11}', 'c_{4}', 'v_{5}'), ('v_{1}', 'c_{3}', 'm_{10}'), ('v_{1}', 'c_{4}', 'm_{10}'), ('v_{3}', 'c_{4}', 'm_{10}'), ('v_{2}', 'c_{3}', 'c_{4}'), ('c_{3}', 'v_{3}', 'm_{10}'), ('c_{3}', 'c_{4}', 'v_{4}'), ('c_{3}', 'v_{4}', 'm_{10}'), ('m_{11}', 'v_{2}', 'c_{4}'), ('m_{11}', 'v_{3}', 'c_{4}'), ('v_{2}', 'c_{3}', 'm_{10}'), ('m_{11}', 'c_{4}', 'v_{4}')], 't_{5}': [('m_{9}', 'v_{7}', 'c_{6}'), ('v_{8}', 'c_{6}', 'm_{8}'), ('v_{6}', 'c_{6}', 'm_{8}'), ('v_{7}', 'c_{5}', 'c_{6}'), ('m_{9}', 'c_{6}', 'v_{9}'), ('m_{9}', 'c_{6}', 'v_{10}'), ('c_{5}', 'c_{6}', 'v_{9}'), ('v_{6}', 'c_{5}', 'c_{6}'), ('m_{9}', 'c_{5}', 'v_{10}'), ('v_{6}', 'c_{5}', 'm_{8}'), ('m_{9}', 'v_{9}', 'm_{8}'), ('v_{7}', 'c_{5}', 'm_{8}'), ('c_{5}', 'v_{8}', 'm_{8}'), ('m_{9}', 'v_{7}', 'm_{8}'), ('c_{5}', 'c_{6}', 'v_{10}'), ('m_{9}', 'c_{5}', 'v_{8}'), ('m_{9}', 'v_{8}', 'c_{6}'), ('c_{5}', 'v_{9}', 'm_{8}')], 's_{5}': [('c_{6}', 'm_{7}', 'v_{10}'), ('c_{6}', 'v_{9}', 'c_{13}'), ('v_{8}', 'm_{7}', 'c_{13}'), ('c_{6}', 'm_{7}', 'v_{11}'), ('c_{6}', 'v_{10}', 'c_{13}'), ('v_{9}', 'm_{7}', 'c_{13}')], 't_{3}': [('m_{11}', 'v_{2}', 'm_{10}'), ('m_{11}', 'c_{8}', 'v_{4}'), ('c_{7}', 'c_{8}', 'v_{5}'), ('v_{3}', 'c_{8}', 'm_{10}'), ('m_{11}', 'v_{4}', 'm_{10}'), ('v_{1}', 'c_{7}', 'm_{10}'), ('v_{2}', 'c_{7}', 'm_{10}'), ('v_{1}', 'c_{8}', 'm_{10}'), ('c_{7}', 'v_{3}', 'm_{10}'), ('m_{11}', 'v_{2}', 'c_{8}'), ('c_{7}', 'c_{8}', 'v_{4}'), ('c_{7}', 'v_{4}', 'm_{10}'), ('m_{11}', 'c_{7}', 'v_{3}'), ('m_{11}', 'v_{3}', 'c_{8}'), ('m_{11}', 'c_{8}', 'v_{5}'), ('v_{1}', 'c_{7}', 'c_{8}'), ('v_{2}', 'c_{7}', 'c_{8}'), ('m_{11}', 'c_{7}', 'v_{5}')], 't_{7}': [('m_{9}', 'c_{10}', 'v_{10}'), ('v_{7}', 'c_{9}', 'm_{8}'), ('c_{9}', 'v_{9}', 'm_{8}'), ('v_{7}', 'c_{9}', 'c_{10}'), ('v_{6}', 'c_{9}', 'c_{10}'), ('m_{9}', 'c_{9}', 'v_{8}'), ('m_{9}', 'c_{10}', 'v_{9}'), ('m_{9}', 'c_{9}', 'v_{10}'), ('c_{9}', 'c_{10}', 'v_{10}'), ('v_{6}', 'c_{10}', 'm_{8}'), ('v_{6}', 'c_{9}', 'm_{8}'), ('v_{8}', 'c_{10}', 'm_{8}'), ('m_{9}', 'v_{9}', 'm_{8}'), ('m_{9}', 'v_{7}', 'c_{10}'), ('m_{9}', 'v_{8}', 'c_{10}'), ('c_{9}', 'v_{8}', 'm_{8}'), ('c_{9}', 'c_{10}', 'v_{9}'), ('m_{9}', 'v_{7}', 'm_{8}')], 's_{7}': [('v_{4}', 'm_{10}', 'm_{12}'), ('v_{3}', 'm_{10}', 'm_{12}'), ('m_{10}', 'm_{12}', 'v_{9}'), ('c_{4}', 'm_{10}', 'v_{9}'), ('m_{10}', 'v_{8}', 'c_{6}'), ('c_{4}', 'm_{10}', 'v_{5}'), ('v_{3}', 'm_{12}', 'c_{6}'), ('c_{4}', 'm_{12}', 'v_{8}'), ('v_{5}', 'm_{12}', 'c_{6}'), ('v_{4}', 'm_{10}', 'c_{6}'), ('c_{4}', 'v_{5}', 'm_{12}'), ('c_{4}', 'v_{8}', 'c_{6}'), ('c_{4}', 'v_{4}', 'c_{6}'), ('c_{4}', 'm_{12}', 'v_{9}'), ('m_{10}', 'm_{12}', 'v_{8}'), ('m_{10}', 'v_{5}', 'c_{6}'), ('v_{3}', 'm_{10}', 'c_{6}'), ('c_{4}', 'v_{4}', 'm_{12}')], 's_{3}': [('c_{15}', 'm_{9}', 'v_{8}'), ('c_{15}', 'v_{7}', 'c_{5}'), ('v_{5}', 'm_{9}', 'c_{5}'), ('c_{15}', 'v_{6}', 'c_{5}'), ('c_{15}', 'm_{9}', 'v_{7}'), ('v_{6}', 'm_{9}', 'c_{5}')], 's_{1}': [('c_{3}', 'v_{6}', 'm_{6}'), ('v_{3}', 'm_{2}', 'c_{5}'), ('m_{4}', 'v_{2}', 'c_{5}'), ('v_{2}', 'c_{3}', 'm_{3}'), ('v_{1}', 'm_{2}', 'c_{5}'), ('v_{2}', 'c_{3}', 'm_{2}'), ('c_{3}', 'v_{3}', 'c_{5}'), ('v_{3}', 'm_{2}', 'm_{6}'), ('v_{1}', 'c_{3}', 'm_{6}'), ('m_{4}', 'c_{5}', 'v_{11}'), ('m_{4}', 'v_{3}', 'm_{2}'), ('c_{3}', 'v_{8}', 'm_{3}'), ('m_{2}', 'c_{5}', 'v_{8}'), ('m_{2}', 'c_{5}', 'v_{11}'), ('m_{4}', 'm_{2}', 'v_{11}'), ('m_{2}', 'v_{6}', 'c_{5}'), ('m_{6}', 'c_{5}', 'v_{8}'), ('v_{2}', 'c_{3}', 'c_{5}'), ('v_{1}', 'm_{6}', 'c_{5}'), ('m_{2}', 'v_{6}', 'm_{3}'), ('m_{4}', 'v_{6}', 'm_{6}'), ('m_{2}', 'm_{6}', 'v_{7}'), ('m_{4}', 'c_{3}', 'v_{3}'), ('m_{4}', 'c_{3}', 'v_{11}'), ('m_{6}', 'v_{8}', 'm_{3}'), ('v_{1}', 'm_{6}', 'm_{3}'), ('c_{3}', 'v_{7}', 'c_{5}'), ('m_{4}', 'v_{2}', 'm_{6}'), ('m_{2}', 'm_{6}', 'v_{11}'), ('c_{3}', 'v_{3}', 'm_{6}'), ('m_{4}', 'v_{2}', 'm_{2}'), ('c_{3}', 'm_{2}', 'v_{6}'), ('m_{4}', 'm_{2}', 'v_{6}'), ('c_{3}', 'm_{2}', 'v_{11}'), ('v_{1}', 'c_{3}', 'm_{3}'), ('m_{6}', 'v_{7}', 'm_{3}'), ('m_{4}', 'c_{5}', 'v_{8}'), ('v_{6}', 'm_{6}', 'm_{3}'), ('m_{4}', 'v_{8}', 'm_{3}'), ('c_{3}', 'c_{5}', 'v_{8}'), ('m_{2}', 'v_{7}', 'c_{5}'), ('v_{7}', 'c_{5}', 'm_{3}'), ('c_{3}', 'c_{5}', 'v_{11}'), ('m_{4}', 'm_{6}', 'v_{11}'), ('m_{4}', 'v_{2}', 'm_{3}'), ('v_{1}', 'm_{2}', 'm_{3}'), ('m_{4}', 'm_{6}', 'v_{7}'), ('v_{2}', 'c_{3}', 'm_{6}'), ('v_{1}', 'c_{3}', 'c_{5}'), ('v_{3}', 'm_{2}', 'm_{3}'), ('v_{1}', 'c_{5}', 'm_{3}'), ('c_{3}', 'm_{6}', 'v_{11}'), ('v_{6}', 'm_{6}', 'c_{5}'), ('c_{3}', 'v_{3}', 'm_{3}'), ('v_{1}', 'c_{3}', 'm_{2}'), ('m_{4}', 'v_{7}', 'c_{5}'), ('m_{2}', 'v_{8}', 'm_{3}'), ('v_{1}', 'm_{2}', 'm_{6}'), ('c_{3}', 'm_{6}', 'v_{7}'), ('m_{6}', 'c_{5}', 'v_{11}')], 'e_{1}': [('c_{15}', 'c_{2}', 'v_{10}'), ('c_{1}', 'c_{2}', 'v_{11}'), ('c_{1}', 'v_{10}', 'c_{13}'), ('c_{1}', 'c_{2}', 'v_{10}'), ('v_{1}', 'c_{15}', 'c_{2}'), ('v_{6}', 'c_{2}', 'c_{13}'), ('c_{15}', 'v_{10}', 'c_{13}'), ('v_{1}', 'c_{2}', 'c_{13}'), ('c_{1}', 'v_{5}', 'c_{2}'), ('c_{1}', 'c_{15}', 'v_{11}'), ('c_{1}', 'v_{6}', 'c_{2}'), ('v_{5}', 'c_{15}', 'c_{13}'), ('c_{15}', 'c_{2}', 'v_{11}'), ('c_{1}', 'v_{5}', 'c_{13}'), ('v_{5}', 'c_{15}', 'c_{2}'), ('v_{1}', 'c_{15}', 'c_{13}'), ('c_{1}', 'c_{15}', 'v_{6}'), ('c_{15}', 'v_{6}', 'c_{13}')], 's_{10}': [('m_{11}', 'c_{7}', 'v_{11}'), ('c_{7}', 'm_{8}', 'v_{11}'), ('v_{1}', 'c_{7}', 'm_{13}'), ('v_{3}', 'm_{13}', 'c_{14}'), ('v_{1}', 'm_{13}', 'm_{8}'), ('v_{1}', 'c_{7}', 'm_{8}'), ('c_{7}', 'm_{8}', 'v_{10}'), ('m_{11}', 'v_{10}', 'c_{14}'), ('m_{11}', 'v_{3}', 'm_{13}'), ('c_{7}', 'v_{10}', 'c_{14}'), ('v_{1}', 'm_{13}', 'c_{14}'), ('c_{7}', 'm_{13}', 'v_{9}'), ('m_{13}', 'm_{8}', 'v_{10}'), ('v_{1}', 'm_{8}', 'c_{14}'), ('m_{11}', 'm_{13}', 'v_{9}'), ('c_{7}', 'v_{3}', 'c_{14}'), ('m_{11}', 'c_{7}', 'v_{3}'), ('m_{11}', 'm_{8}', 'v_{10}'), ('v_{2}', 'c_{7}', 'm_{8}'), ('c_{7}', 'v_{9}', 'm_{8}'), ('m_{11}', 'm_{13}', 'v_{11}'), ('v_{1}', 'c_{7}', 'c_{14}'), ('v_{2}', 'c_{7}', 'm_{13}'), ('v_{3}', 'm_{13}', 'm_{8}'), ('m_{11}', 'v_{9}', 'm_{8}'), ('c_{7}', 'm_{13}', 'v_{11}'), ('m_{13}', 'v_{9}', 'c_{14}'), ('v_{9}', 'm_{8}', 'c_{14}'), ('m_{13}', 'm_{8}', 'v_{11}'), ('m_{11}', 'v_{2}', 'c_{14}'), ('v_{2}', 'c_{7}', 'c_{14}'), ('m_{11}', 'v_{2}', 'm_{8}'), ('m_{11}', 'v_{2}', 'm_{13}'), ('m_{11}', 'm_{8}', 'v_{11}'), ('m_{13}', 'v_{10}', 'c_{14}'), ('c_{7}', 'v_{3}', 'm_{8}')], 't_{8}': [('m_{6}', 'v_{7}', 'm_{7}'), ('v_{8}', 'c_{10}', 'm_{7}'), ('v_{7}', 'c_{9}', 'c_{10}'), ('m_{6}', 'v_{7}', 'c_{10}'), ('v_{6}', 'c_{9}', 'm_{7}'), ('m_{6}', 'c_{10}', 'v_{9}'), ('m_{6}', 'c_{10}', 'v_{10}'), ('c_{9}', 'c_{10}', 'v_{10}'), ('m_{6}', 'v_{8}', 'c_{10}'), ('v_{6}', 'c_{9}', 'c_{10}'), ('v_{6}', 'c_{10}', 'm_{7}'), ('c_{9}', 'v_{9}', 'm_{7}'), ('m_{6}', 'v_{9}', 'm_{7}'), ('m_{6}', 'c_{9}', 'v_{10}'), ('c_{9}', 'v_{8}', 'm_{7}'), ('v_{7}', 'c_{9}', 'm_{7}'), ('c_{9}', 'c_{10}', 'v_{9}'), ('m_{6}', 'c_{9}', 'v_{8}')], 'q_{1}': [('v_{2}', 'c_{3}', 's_{7}'), ('c_{4}', 'm_{12}', 'm_{8}'), ('c_{1}', 'c_{15}', 'c_{5}'), ('c_{1}', 'm_{9}', 'c_{6}'), ('m_{12}', 'c_{6}', 'm_{8}'), ('v_{5}', 's_{3}', 'm_{8}'), ('m_{10}', 'c_{6}', 'c_{13}'), ('m_{10}', 'm_{12}', 'm_{8}'), ('c_{15}', 'c_{5}', 'c_{13}'), ('m_{11}', 'c_{4}', 'm_{10}'), ('m_{10}', 'c_{6}', 'm_{8}'), ('v_{5}', 's_{3}', 'c_{13}'), ('c_{1}', 'c_{5}', 'c_{6}'), ('c_{1}', 'c_{6}', 'm_{8}'), ('c_{4}', 'c_{6}', 'c_{13}'), ('c_{15}', 'm_{9}', 'c_{5}'), ('v_{1}', 's_{3}', 'm_{8}'), ('t_{1}', 'c_{6}', 'v_{11}'), ('c_{3}', 's_{7}', 'v_{9}'), ('c_{15}', 'c_{6}', 'm_{8}'), ('v_{3}', 's_{7}', 'm_{8}'), ('c_{15}', 't_{5}', 'v_{11}'), ('s_{7}', 'v_{9}', 'c_{13}'), ('c_{1}', 's_{3}', 'v_{8}'), ('m_{11}', 's_{7}', 'v_{9}'), ('c_{3}', 'm_{12}', 'c_{6}'), ('v_{1}', 't_{5}', 'c_{13}'), ('t_{1}', 'v_{5}', 'c_{13}'), ('m_{11}', 'v_{2}', 's_{7}'), ('t_{1}', 'm_{8}', 'v_{11}'), ('s_{3}', 'c_{6}', 'v_{9}'), ('s_{3}', 'c_{6}', 'v_{11}'), ('s_{3}', 'v_{8}', 'm_{8}'), ('m_{11}', 's_{7}', 'v_{11}'), ('t_{1}', 'm_{12}', 'v_{8}'), ('m_{11}', 'm_{12}', 'c_{6}'), ('m_{10}', 'm_{12}', 'c_{13}'), ('t_{1}', 'm_{12}', 'v_{11}'), ('m_{12}', 'c_{6}', 'c_{13}'), ('c_{3}', 's_{7}', 'v_{11}'), ('t_{1}', 'c_{6}', 'v_{9}'), ('v_{5}', 's_{3}', 'c_{6}'), ('c_{1}', 'c_{5}', 'm_{8}'), ('v_{1}', 's_{3}', 'c_{6}'), ('c_{3}', 'm_{10}', 'm_{8}'), ('c_{4}', 'm_{12}', 'c_{13}'), ('c_{6}', 'm_{8}', 'c_{13}'), ('c_{4}', 'm_{10}', 'm_{12}'), ('s_{3}', 'm_{8}', 'v_{10}'), ('v_{1}', 's_{7}', 'm_{8}'), ('m_{11}', 'c_{4}', 'c_{13}'), ('m_{9}', 'c_{5}', 'c_{6}'), ('t_{1}', 'v_{5}', 'c_{6}'), ('c_{1}', 's_{3}', 'v_{11}'), ('t_{1}', 'm_{8}', 'v_{10}'), ('m_{11}', 'c_{3}', 'm_{8}'), ('m_{11}', 'c_{3}', 'c_{13}'), ('m_{9}', 'c_{6}', 'c_{13}'), ('v_{1}', 's_{3}', 'c_{13}'), ('s_{7}', 'm_{8}', 'v_{11}'), ('c_{1}', 't_{5}', 'v_{10}'), ('c_{3}', 'c_{4}', 'm_{10}'), ('s_{7}', 'v_{10}', 'c_{13}'), ('s_{3}', 'v_{8}', 'c_{13}'), ('c_{15}', 'm_{9}', 'c_{13}'), ('m_{9}', 'c_{5}', 'm_{8}'), ('s_{3}', 'v_{10}', 'c_{13}'), ('m_{11}', 'c_{3}', 'm_{12}'), ('m_{11}', 'c_{3}', 'c_{6}'), ('t_{1}', 'v_{9}', 'm_{8}'), ('v_{1}', 'c_{3}', 's_{7}'), ('s_{3}', 'v_{9}', 'm_{8}'), ('v_{3}', 's_{7}', 'c_{13}'), ('t_{1}', 'v_{5}', 'm_{8}'), ('c_{4}', 'c_{6}', 'm_{8}'), ('s_{3}', 'm_{8}', 'v_{11}'), ('s_{7}', 'm_{8}', 'v_{10}'), ('m_{11}', 'c_{4}', 'm_{8}'), ('c_{1}', 'm_{9}', 'm_{8}'), ('c_{5}', 'm_{8}', 'c_{13}'), ('m_{11}', 'm_{10}', 'c_{13}'), ('t_{1}', 'v_{8}', 'c_{6}'), ('c_{3}', 'c_{4}', 'c_{13}'), ('v_{5}', 'c_{15}', 't_{5}'), ('c_{3}', 'm_{10}', 'c_{13}'), ('c_{15}', 't_{5}', 'v_{10}'), ('c_{1}', 't_{5}', 'v_{11}'), ('c_{3}', 'c_{4}', 'm_{8}'), ('v_{6}', 't_{5}', 'c_{13}'), ('c_{5}', 'c_{6}', 'c_{13}'), ('m_{11}', 'm_{10}', 'm_{8}'), ('c_{1}', 'v_{6}', 't_{5}'), ('c_{4}', 'm_{10}', 'c_{6}'), ('v_{1}', 'c_{15}', 't_{5}'), ('t_{1}', 'v_{10}', 'c_{13}'), ('c_{1}', 'v_{5}', 't_{5}'), ('c_{1}', 'c_{15}', 'm_{9}'), ('v_{1}', 's_{7}', 'c_{13}'), ('m_{11}', 'v_{3}', 's_{7}'), ('m_{9}', 'm_{8}', 'c_{13}')], 's_{9}': [('c_{3}', 'm_{8}', 'v_{11}'), ('m_{13}', 'm_{8}', 'v_{10}'), ('c_{3}', 'v_{10}', 'c_{13}'), ('v_{1}', 'm_{13}', 'm_{8}'), ('v_{2}', 'c_{3}', 'c_{13}'), ('m_{13}', 'v_{10}', 'c_{13}'), ('m_{11}', 'v_{10}', 'c_{13}'), ('c_{3}', 'm_{13}', 'v_{9}'), ('m_{11}', 'v_{3}', 'm_{13}'), ('v_{1}', 'm_{13}', 'c_{13}'), ('v_{2}', 'c_{3}', 'm_{13}'), ('m_{11}', 'm_{13}', 'v_{9}'), ('c_{3}', 'v_{3}', 'm_{8}'), ('c_{3}', 'v_{3}', 'c_{13}'), ('c_{3}', 'v_{9}', 'm_{8}'), ('m_{11}', 'm_{8}', 'v_{10}'), ('v_{1}', 'm_{8}', 'c_{13}'), ('m_{11}', 'm_{13}', 'v_{11}'), ('v_{3}', 'm_{13}', 'm_{8}'), ('m_{11}', 'v_{9}', 'm_{8}'), ('c_{3}', 'm_{8}', 'v_{10}'), ('m_{11}', 'v_{2}', 'c_{13}'), ('c_{3}', 'm_{13}', 'v_{11}'), ('v_{9}', 'm_{8}', 'c_{13}'), ('v_{1}', 'c_{3}', 'm_{13}'), ('m_{13}', 'm_{8}', 'v_{11}'), ('m_{13}', 'v_{9}', 'c_{13}'), ('v_{1}', 'c_{3}', 'c_{13}'), ('v_{1}', 'c_{3}', 'm_{8}'), ('m_{11}', 'v_{2}', 'm_{8}'), ('v_{3}', 'm_{13}', 'c_{13}'), ('m_{11}', 'c_{3}', 'v_{11}'), ('m_{11}', 'v_{2}', 'm_{13}'), ('m_{11}', 'c_{3}', 'v_{3}'), ('m_{11}', 'm_{8}', 'v_{11}'), ('v_{2}', 'c_{3}', 'm_{8}')], 's_{12}': [('v_{3}', 'm_{5}', 'c_{16}'), ('c_{8}', 'm_{5}', 'v_{6}'), ('c_{8}', 'm_{5}', 'v_{5}'), ('c_{8}', 'v_{5}', 'c_{16}'), ('c_{8}', 'v_{4}', 'c_{16}'), ('v_{4}', 'm_{5}', 'c_{16}')]}
    # BR_delta_C3 = integrate_BR(BR_nabla_delta_C3)
    # BR_delta_C3 = chain_map_mod(expand_map_all(BR_delta_C3))
    # print "\nintegrate(nabla Delta_C3) =", BR_delta_C3
    # BR_nabla_delta_c3_computed = derivative(BR_delta_C3, BR_C)
    # BR_nabla_delta_c3_computed = chain_map_mod(expand_map_all(BR_nabla_delta_c3_computed))
    # print "(1 x Delta_C + Delta_C x 1) Delta_C + Nabla(Delta_C3) = 0 ? ",
    # print not any(add_maps_mod_2(BR_nabla_delta_C3, BR_nabla_delta_c3_computed).values())

    BR_min_remainder = [('m_{9}', 'c_{7}', 'm_{2}'), ('s_{10}', 'm_{7}', 'v_{10}'), ('m_{9}', 's_{9}', 'v_{11}'), ('m_{6}', 'm_{11}', 'c_{13}'), ('m_{11}', 'c_{13}', 'm_{8}'), ('m_{9}', 'c_{10}', 'm_{13}'), ('m_{9}', 'v_{8}', 's_{6}'), ('m_{6}', 'c_{5}', 'c_{6}'), ('m_{6}', 'c_{3}', 'c_{13}'), ('v_{8}', 's_{6}', 'm_{7}'), ('c_{7}', 'm_{13}', 'm_{7}'), ('m_{6}', 'c_{6}', 'm_{7}'), ('m_{9}', 'c_{7}', 'm_{13}'), ('m_{4}', 'c_{10}', 'm_{8}'), ('m_{8}', 'c_{13}', 'm_{7}'), ('m_{9}', 's_{10}', 'v_{3}'), ('m_{6}', 's_{10}', 'v_{11}'), ('c_{3}', 'm_{2}', 'm_{8}'), ('m_{9}', 's_{1}', 'v_{8}'), ('m_{6}', 'v_{7}', 's_{5}'), ('m_{9}', 'm_{4}', 'c_{5}'), ('m_{6}', 'c_{7}', 'm_{2}'), ('m_{6}', 'm_{11}', 'c_{14}'), ('m_{6}', 's_{2}', 'v_{3}'), ('m_{9}', 'm_{7}', 'c_{13}'), ('m_{9}', 'm_{11}', 'c_{14}'), ('s_{2}', 'v_{8}', 'm_{8}'), ('m_{9}', 'm_{6}', 'c_{5}'), ('c_{3}', 'c_{6}', 'm_{7}'), ('s_{10}', 'v_{11}', 'm_{8}'), ('m_{13}', 'c_{14}', 'm_{8}'), ('s_{2}', 'm_{8}', 'v_{9}'), ('m_{9}', 's_{1}', 'v_{3}'), ('m_{6}', 'm_{8}', 'c_{14}'), ('m_{9}', 'v_{7}', 's_{1}'), ('m_{6}', 'c_{3}', 'c_{5}'), ('s_{10}', 'm_{7}', 'v_{9}'), ('m_{9}', 'c_{7}', 'c_{9}'), ('m_{6}', 's_{1}', 'v_{8}'), ('c_{7}', 'c_{9}', 'm_{7}'), ('c_{3}', 'c_{9}', 'm_{8}'), ('m_{6}', 'c_{6}', 'c_{13}'), ('s_{9}', 'v_{9}', 'm_{7}'), ('s_{6}', 'v_{9}', 'm_{8}'), ('m_{6}', 'v_{7}', 's_{6}'), ('m_{6}', 's_{5}', 'v_{11}'), ('m_{6}', 'v_{7}', 's_{2}'), ('m_{9}', 'c_{7}', 'm_{8}'), ('m_{4}', 'c_{6}', 'm_{7}'), ('s_{9}', 'm_{8}', 'v_{9}'), ('m_{6}', 'v_{8}', 's_{6}'), ('m_{9}', 's_{6}', 'v_{3}'), ('m_{9}', 'm_{6}', 'c_{9}'), ('m_{11}', 'c_{13}', 'm_{7}'), ('m_{6}', 'm_{7}', 'c_{14}'), ('c_{6}', 'm_{7}', 'm_{8}'), ('c_{3}', 'm_{2}', 'm_{7}'), ('m_{8}', 'c_{14}', 'm_{8}'), ('m_{6}', 'c_{9}', 'c_{10}'), ('c_{3}', 'c_{10}', 'm_{7}'), ('m_{6}', 'v_{7}', 's_{1}'), ('s_{2}', 'v_{9}', 'm_{8}'), ('m_{6}', 's_{1}', 'v_{3}'), ('m_{11}', 'c_{14}', 'm_{7}'), ('m_{9}', 'c_{5}', 'm_{8}'), ('m_{6}', 'c_{5}', 'm_{13}'), ('m_{9}', 'v_{7}', 's_{10}'), ('m_{6}', 'c_{5}', 'm_{8}'), ('m_{9}', 'm_{9}', 'c_{7}'), ('c_{5}', 'm_{8}', 'm_{7}'), ('m_{6}', 's_{5}', 'v_{3}'), ('m_{9}', 's_{9}', 'v_{3}'), ('m_{6}', 'm_{2}', 'c_{5}'), ('m_{6}', 'm_{9}', 'c_{3}'), ('c_{9}', 'c_{10}', 'm_{8}'), ('s_{2}', 'v_{9}', 'm_{7}'), ('m_{9}', 's_{6}', 'v_{11}'), ('m_{6}', 'c_{6}', 'm_{13}'), ('m_{9}', 'c_{3}', 'm_{8}'), ('c_{3}', 'c_{6}', 'm_{8}'), ('c_{3}', 'm_{8}', 'm_{7}'), ('s_{1}', 'v_{8}', 'm_{7}'), ('m_{9}', 'c_{3}', 'm_{13}'), ('m_{9}', 'c_{9}', 'c_{10}'), ('c_{3}', 'c_{10}', 'm_{8}'), ('m_{9}', 'c_{3}', 'm_{6}'), ('m_{7}', 'c_{14}', 'm_{8}'), ('m_{8}', 'c_{13}', 'm_{8}'), ('m_{9}', 'm_{6}', 'c_{7}'), ('m_{9}', 'm_{7}', 'c_{14}'), ('m_{6}', 'v_{7}', 's_{9}'), ('m_{9}', 'v_{7}', 's_{5}'), ('c_{5}', 'c_{6}', 'm_{7}'), ('m_{13}', 'c_{14}', 'm_{7}'), ('m_{9}', 'c_{5}', 'c_{6}'), ('m_{9}', 's_{2}', 'v_{3}'), ('c_{3}', 'm_{13}', 'm_{7}'), ('m_{6}', 'v_{8}', 's_{5}'), ('m_{9}', 'c_{3}', 'c_{13}'), ('c_{10}', 'm_{7}', 'm_{8}'), ('c_{9}', 'm_{7}', 'm_{7}'), ('m_{4}', 'c_{6}', 'm_{8}'), ('s_{9}', 'v_{11}', 'm_{8}'), ('s_{5}', 'v_{11}', 'm_{8}'), ('s_{6}', 'v_{11}', 'm_{7}'), ('c_{7}', 'c_{14}', 'm_{7}'), ('m_{9}', 'v_{7}', 's_{6}'), ('m_{9}', 'm_{2}', 'c_{9}'), ('s_{1}', 'm_{7}', 'v_{9}'), ('m_{13}', 'c_{13}', 'm_{7}'), ('m_{9}', 'v_{1}', 's_{1}'), ('m_{6}', 'm_{6}', 'c_{9}'), ('m_{6}', 's_{9}', 'v_{3}'), ('m_{6}', 'v_{1}', 's_{1}'), ('c_{7}', 'c_{9}', 'm_{8}'), ('s_{1}', 'm_{8}', 'v_{10}'), ('v_{8}', 's_{5}', 'm_{8}'), ('s_{9}', 'm_{8}', 'v_{10}'), ('c_{6}', 'c_{13}', 'm_{7}'), ('v_{1}', 's_{6}', 'm_{8}'), ('c_{10}', 'c_{14}', 'm_{7}'), ('v_{1}', 's_{6}', 'm_{7}'), ('m_{9}', 'm_{8}', 'c_{14}'), ('m_{6}', 's_{6}', 'v_{11}'), ('m_{6}', 'm_{6}', 'c_{7}'), ('c_{10}', 'm_{8}', 'm_{7}'), ('m_{2}', 'c_{10}', 'm_{8}'), ('m_{9}', 'm_{2}', 'c_{5}'), ('m_{9}', 'm_{9}', 'c_{3}'), ('m_{9}', 'c_{7}', 'c_{14}'), ('c_{7}', 'm_{6}', 'm_{7}'), ('s_{5}', 'v_{9}', 'm_{8}'), ('m_{9}', 'c_{5}', 'm_{13}'), ('s_{10}', 'm_{8}', 'v_{10}'), ('s_{6}', 'v_{11}', 'm_{8}'), ('v_{8}', 's_{5}', 'm_{7}'), ('c_{6}', 'c_{13}', 'm_{8}'), ('m_{7}', 'c_{13}', 'm_{8}'), ('m_{6}', 'v_{1}', 's_{9}'), ('m_{6}', 'm_{7}', 'c_{13}'), ('s_{1}', 'v_{9}', 'm_{8}'), ('m_{6}', 'm_{13}', 'c_{13}'), ('s_{9}', 'v_{11}', 'm_{7}'), ('s_{9}', 'v_{9}', 'm_{8}'), ('m_{6}', 'c_{3}', 'm_{6}'), ('m_{6}', 'c_{9}', 'm_{7}'), ('m_{9}', 'c_{9}', 'm_{7}'), ('c_{7}', 'm_{8}', 'm_{7}'), ('s_{2}', 'm_{7}', 'v_{9}'), ('m_{13}', 'c_{13}', 'm_{8}'), ('m_{9}', 'm_{4}', 'c_{9}'), ('m_{6}', 'c_{3}', 'm_{2}'), ('m_{6}', 'm_{4}', 'c_{9}'), ('m_{9}', 'v_{1}', 's_{2}'), ('m_{2}', 'c_{10}', 'm_{7}'), ('m_{6}', 'm_{6}', 'c_{3}'), ('m_{7}', 'c_{14}', 'm_{7}'), ('c_{10}', 'c_{14}', 'm_{8}'), ('m_{6}', 'm_{2}', 'c_{9}'), ('m_{9}', 'c_{6}', 'm_{13}'), ('s_{5}', 'm_{7}', 'v_{10}'), ('m_{9}', 'c_{7}', 'm_{6}'), ('m_{6}', 'v_{7}', 's_{10}'), ('m_{6}', 'c_{3}', 'm_{13}'), ('c_{3}', 'm_{13}', 'm_{8}'), ('m_{9}', 'c_{9}', 'm_{8}'), ('c_{7}', 'm_{2}', 'm_{8}'), ('m_{6}', 'c_{3}', 'm_{8}'), ('m_{9}', 'c_{10}', 'm_{8}'), ('m_{6}', 'c_{9}', 'm_{13}'), ('s_{5}', 'v_{9}', 'm_{7}'), ('m_{9}', 's_{5}', 'v_{11}'), ('m_{9}', 's_{2}', 'v_{8}'), ('v_{1}', 's_{5}', 'm_{7}'), ('s_{5}', 'v_{11}', 'm_{7}'), ('m_{6}', 'c_{9}', 'm_{8}'), ('c_{3}', 'c_{9}', 'm_{7}'), ('c_{6}', 'm_{8}', 'm_{7}'), ('c_{9}', 'c_{10}', 'm_{7}'), ('m_{6}', 's_{10}', 'v_{3}'), ('m_{9}', 'v_{8}', 's_{5}'), ('s_{10}', 'm_{8}', 'v_{9}'), ('m_{11}', 'c_{14}', 'm_{8}'), ('m_{6}', 'c_{7}', 'c_{9}'), ('c_{3}', 'c_{13}', 'm_{8}'), ('m_{6}', 'c_{7}', 'm_{6}'), ('c_{7}', 'm_{2}', 'm_{7}'), ('m_{6}', 's_{6}', 'v_{3}'), ('m_{9}', 'm_{13}', 'c_{13}'), ('s_{5}', 'm_{8}', 'v_{10}'), ('m_{9}', 'v_{1}', 's_{10}'), ('m_{6}', 'v_{1}', 's_{10}'), ('m_{9}', 'm_{13}', 'c_{14}'), ('c_{7}', 'm_{6}', 'm_{8}'), ('m_{4}', 'c_{10}', 'm_{7}'), ('m_{9}', 'v_{1}', 's_{9}'), ('v_{1}', 's_{5}', 'm_{8}'), ('m_{6}', 's_{9}', 'v_{11}'), ('s_{2}', 'm_{7}', 'v_{10}'), ('s_{6}', 'm_{8}', 'v_{9}'), ('m_{2}', 'c_{6}', 'm_{7}'), ('s_{6}', 'm_{7}', 'v_{10}'), ('c_{5}', 'm_{7}', 'm_{7}'), ('m_{9}', 'c_{5}', 'm_{7}'), ('m_{9}', 's_{5}', 'v_{3}'), ('c_{3}', 'c_{13}', 'm_{7}'), ('s_{1}', 'm_{7}', 'v_{10}'), ('m_{9}', 'c_{3}', 'm_{2}'), ('m_{8}', 'c_{14}', 'm_{7}'), ('m_{6}', 'c_{10}', 'c_{14}'), ('m_{9}', 'c_{6}', 'c_{13}'), ('m_{6}', 'c_{10}', 'm_{13}'), ('m_{6}', 'v_{1}', 's_{2}'), ('c_{3}', 'm_{8}', 'm_{8}'), ('s_{6}', 'm_{7}', 'v_{9}'), ('m_{2}', 'c_{6}', 'm_{8}'), ('s_{5}', 'm_{8}', 'v_{9}'), ('m_{6}', 'm_{6}', 'c_{5}'), ('s_{1}', 'v_{8}', 'm_{8}'), ('s_{6}', 'v_{9}', 'm_{7}'), ('c_{7}', 'c_{14}', 'm_{8}'), ('m_{6}', 'c_{10}', 'm_{7}'), ('s_{6}', 'm_{8}', 'v_{10}'), ('m_{9}', 'v_{7}', 's_{9}'), ('m_{6}', 'c_{7}', 'c_{14}'), ('m_{6}', 'c_{5}', 'm_{7}'), ('m_{9}', 'c_{3}', 'c_{5}'), ('s_{5}', 'm_{7}', 'v_{9}'), ('m_{6}', 'm_{9}', 'c_{7}'), ('s_{9}', 'm_{7}', 'v_{10}'), ('s_{1}', 'm_{8}', 'v_{9}'), ('s_{2}', 'v_{8}', 'm_{7}'), ('m_{6}', 'm_{13}', 'c_{14}'), ('m_{6}', 'm_{4}', 'c_{5}'), ('m_{9}', 'm_{6}', 'c_{3}'), ('s_{1}', 'v_{9}', 'm_{7}'), ('s_{9}', 'm_{7}', 'v_{9}'), ('m_{9}', 'c_{9}', 'm_{13}'), ('c_{7}', 'm_{13}', 'm_{8}'), ('m_{9}', 'c_{10}', 'c_{14}'), ('m_{6}', 'c_{7}', 'm_{8}'), ('s_{10}', 'v_{9}', 'm_{8}'), ('s_{2}', 'm_{8}', 'v_{10}'), ('m_{7}', 'c_{13}', 'm_{7}'), ('s_{10}', 'v_{11}', 'm_{7}'), ('m_{9}', 'm_{11}', 'c_{13}'), ('m_{6}', 'm_{8}', 'c_{13}'), ('c_{9}', 'm_{8}', 'm_{7}'), ('m_{6}', 's_{2}', 'v_{8}'), ('m_{9}', 's_{10}', 'v_{11}'), ('c_{3}', 'm_{6}', 'm_{8}'), ('c_{7}', 'm_{8}', 'm_{8}'), ('v_{8}', 's_{6}', 'm_{8}'), ('m_{6}', 'c_{7}', 'm_{13}'), ('s_{10}', 'v_{9}', 'm_{7}'), ('m_{9}', 'v_{7}', 's_{2}'), ('c_{3}', 'm_{6}', 'm_{7}'), ('c_{5}', 'c_{6}', 'm_{8}'), ('m_{9}', 'm_{8}', 'c_{13}'), ('m_{9}', 'c_{6}', 'm_{8}')]
    BR_min_remainder_factored = factorize_cycles(BR_min_remainder, BR_C)
    print "\nBR_min_remainder =", BR_min_remainder_factored

    BR_min_remainder_fd = [tuple(map(f_BR, list(t))) for t in BR_min_remainder_factored]
    print "\nf(BR_min_remainder) =", BR_min_remainder_fd

    # test group_boundary_chains
    BR_test_chains1 = [(['m_{11}', 'm_{4}'], ['m_{4}'], ['v_{2}']), (['m_{11}', 'm_{4}'], ['m_{4}'], ['v_{1}']), (['m_{11}', 'm_{4}'], ['v_{2}'], ['m_{4}']), (['m_{11}', 'm_{4}'], ['v_{1}'], ['m_{4}'])]
    print "grouped chains:", group_boundary_chains(BR_test_chains1, BR_C)

    # test derivation of Delta_3, g^3
    BR_phi_1 = {'h0_0': [], 'h2_1': [('v_{1}', 't_{5}', 'm_{4}'), ('v_{1}', 't_{5}', 'm_{7}'), ('v_{1}', 't_{5}', 'm_{13}'), ('v_{1}', 't_{5}', 'c_{3}'), ('v_{1}', 't_{6}', 'm_{4}'), ('v_{1}', 't_{6}', 'm_{7}'), ('v_{1}', 't_{6}', 'm_{13}'), ('v_{1}', 't_{6}', 'c_{3}'), ('v_{1}', 't_{7}', 'm_{4}'), ('v_{1}', 't_{7}', 'm_{7}'), ('v_{1}', 't_{7}', 'm_{13}'), ('v_{1}', 't_{7}', 'c_{3}'), ('v_{1}', 't_{8}', 'm_{4}'), ('v_{1}', 't_{8}', 'm_{7}'), ('v_{1}', 't_{8}', 'm_{13}'), ('v_{1}', 't_{8}', 'c_{3}'), ('v_{1}', 's_{5}', 'm_{7}'), ('v_{1}', 's_{5}', 'm_{8}'), ('v_{1}', 's_{6}', 'm_{7}'), ('v_{1}', 's_{6}', 'm_{8}'), ('v_{1}', 'm_{6}', 's_{1}'), ('v_{1}', 'm_{6}', 's_{2}'), ('v_{1}', 'm_{6}', 's_{5}'), ('v_{1}', 'm_{6}', 's_{6}'), ('v_{1}', 'm_{6}', 's_{9}'), ('v_{1}', 'm_{6}', 's_{10}'), ('v_{1}', 'm_{9}', 's_{1}'), ('v_{1}', 'm_{9}', 's_{2}'), ('v_{1}', 'm_{9}', 's_{5}'), ('v_{1}', 'm_{9}', 's_{6}'), ('v_{1}', 'm_{9}', 's_{9}'), ('v_{1}', 'm_{9}', 's_{10}'), ('v_{1}', 'm_{2}', 't_{5}'), ('v_{1}', 'm_{2}', 't_{6}'), ('v_{1}', 'm_{2}', 't_{7}'), ('v_{1}', 'm_{2}', 't_{8}'), ('v_{1}', 'c_{3}', 't_{5}'), ('v_{1}', 'c_{3}', 't_{6}'), ('v_{1}', 'c_{3}', 't_{7}'), ('v_{1}', 'c_{3}', 't_{8}'), ('t_{5}', 'm_{4}', 'v_{1}'), ('t_{5}', 'm_{7}', 'v_{1}'), ('t_{5}', 'm_{13}', 'v_{1}'), ('t_{5}', 'c_{3}', 'v_{1}'), ('t_{6}', 'm_{4}', 'v_{1}'), ('t_{6}', 'm_{7}', 'v_{1}'), ('t_{6}', 'm_{13}', 'v_{1}'), ('t_{6}', 'c_{3}', 'v_{1}'), ('t_{7}', 'm_{4}', 'v_{1}'), ('t_{7}', 'm_{7}', 'v_{1}'), ('t_{7}', 'm_{13}', 'v_{1}'), ('t_{7}', 'c_{3}', 'v_{1}'), ('t_{8}', 'm_{4}', 'v_{1}'), ('t_{8}', 'm_{7}', 'v_{1}'), ('t_{8}', 'm_{13}', 'v_{1}'), ('t_{8}', 'c_{3}', 'v_{1}'), ('s_{1}', 'm_{7}', 'v_{1}'), ('s_{1}', 'm_{8}', 'v_{1}'), ('s_{2}', 'm_{7}', 'v_{1}'), ('s_{2}', 'm_{8}', 'v_{1}'), ('s_{5}', 'm_{7}', 'v_{1}'), ('s_{5}', 'm_{8}', 'v_{1}'), ('s_{6}', 'm_{7}', 'v_{1}'), ('s_{6}', 'm_{8}', 'v_{1}'), ('s_{9}', 'm_{7}', 'v_{1}'), ('s_{9}', 'm_{8}', 'v_{1}'), ('s_{10}', 'm_{7}', 'v_{1}'), ('s_{10}', 'm_{8}', 'v_{1}'), ('m_{6}', 's_{1}', 'v_{1}'), ('m_{6}', 's_{2}', 'v_{1}'), ('m_{6}', 's_{5}', 'v_{1}'), ('m_{6}', 's_{6}', 'v_{1}'), ('m_{6}', 's_{9}', 'v_{1}'), ('m_{6}', 's_{10}', 'v_{1}'), ('m_{9}', 's_{1}', 'v_{1}'), ('m_{9}', 's_{2}', 'v_{1}'), ('m_{9}', 's_{5}', 'v_{1}'), ('m_{9}', 's_{6}', 'v_{1}'), ('m_{9}', 's_{9}', 'v_{1}'), ('m_{9}', 's_{10}', 'v_{1}'), ('m_{2}', 't_{5}', 'v_{1}'), ('m_{2}', 't_{6}', 'v_{1}'), ('m_{2}', 't_{7}', 'v_{1}'), ('m_{2}', 't_{8}', 'v_{1}'), ('m_{4}', 't_{5}', 'v_{1}'), ('m_{4}', 't_{6}', 'v_{1}'), ('m_{4}', 't_{7}', 'v_{1}'), ('m_{4}', 't_{8}', 'v_{1}'), ('c_{3}', 't_{5}', 'v_{1}'), ('c_{3}', 't_{6}', 'v_{1}'), ('c_{3}', 't_{7}', 'v_{1}'), ('c_{3}', 't_{8}', 'v_{1}'), ('t_{5}', 'v_{1}', 'm_{4}'), ('t_{5}', 'm_{4}', 'v_{2}'), ('t_{5}', 'm_{7}', 'v_{10}'), ('t_{5}', 'v_{9}', 'm_{7}'), ('t_{5}', 'm_{13}', 'v_{9}'), ('t_{5}', 'v_{3}', 'm_{13}'), ('t_{5}', 'v_{2}', 'c_{3}'), ('t_{5}', 'c_{3}', 'v_{3}'), ('t_{6}', 'v_{1}', 'm_{4}'), ('t_{6}', 'm_{4}', 'v_{2}'), ('t_{6}', 'm_{7}', 'v_{10}'), ('t_{6}', 'v_{9}', 'm_{7}'), ('t_{6}', 'm_{13}', 'v_{9}'), ('t_{6}', 'v_{3}', 'm_{13}'), ('t_{6}', 'v_{2}', 'c_{3}'), ('t_{6}', 'c_{3}', 'v_{3}'), ('t_{7}', 'v_{1}', 'm_{4}'), ('t_{7}', 'm_{4}', 'v_{2}'), ('t_{7}', 'm_{7}', 'v_{10}'), ('t_{7}', 'v_{9}', 'm_{7}'), ('t_{7}', 'm_{13}', 'v_{9}'), ('t_{7}', 'v_{3}', 'm_{13}'), ('t_{7}', 'v_{2}', 'c_{3}'), ('t_{7}', 'c_{3}', 'v_{3}'), ('t_{8}', 'v_{1}', 'm_{4}'), ('t_{8}', 'm_{4}', 'v_{2}'), ('t_{8}', 'm_{7}', 'v_{10}'), ('t_{8}', 'v_{9}', 'm_{7}'), ('t_{8}', 'm_{13}', 'v_{9}'), ('t_{8}', 'v_{3}', 'm_{13}'), ('t_{8}', 'v_{2}', 'c_{3}'), ('t_{8}', 'c_{3}', 'v_{3}'), ('s_{1}', 'm_{7}', 'v_{10}'), ('s_{1}', 'v_{9}', 'm_{7}'), ('s_{1}', 'v_{9}', 'm_{8}'), ('s_{1}', 'm_{8}', 'v_{10}'), ('s_{2}', 'm_{7}', 'v_{10}'), ('s_{2}', 'v_{9}', 'm_{7}'), ('s_{2}', 'v_{9}', 'm_{8}'), ('s_{2}', 'm_{8}', 'v_{10}'), ('s_{5}', 'm_{7}', 'v_{10}'), ('s_{5}', 'v_{9}', 'm_{7}'), ('s_{5}', 'v_{9}', 'm_{8}'), ('s_{5}', 'm_{8}', 'v_{10}'), ('s_{6}', 'm_{7}', 'v_{10}'), ('s_{6}', 'v_{9}', 'm_{7}'), ('s_{6}', 'v_{9}', 'm_{8}'), ('s_{6}', 'm_{8}', 'v_{10}'), ('s_{9}', 'm_{7}', 'v_{10}'), ('s_{9}', 'v_{9}', 'm_{7}'), ('s_{9}', 'v_{9}', 'm_{8}'), ('s_{9}', 'm_{8}', 'v_{10}'), ('s_{10}', 'm_{7}', 'v_{10}'), ('s_{10}', 'v_{9}', 'm_{7}'), ('s_{10}', 'v_{9}', 'm_{8}'), ('s_{10}', 'm_{8}', 'v_{10}'), ('m_{6}', 'c_{3}', 'm_{3}'), ('m_{6}', 'c_{5}', 'm_{3}'), ('m_{6}', 'v_{1}', 's_{1}'), ('m_{6}', 'm_{4}', 'c_{3}'), ('m_{6}', 'c_{3}', 'c_{5}'), ('m_{6}', 'm_{2}', 'c_{5}'), ('m_{6}', 'c_{3}', 'm_{6}'), ('m_{6}', 'm_{4}', 'c_{5}'), ('m_{6}', 'c_{3}', 'm_{2}'), ('m_{6}', 's_{1}', 'v_{11}'), ('m_{6}', 'm_{6}', 'c_{5}'), ('m_{6}', 'c_{7}', 'm_{6}'), ('m_{6}', 'v_{1}', 's_{2}'), ('m_{6}', 'c_{7}', 'c_{9}'), ('m_{6}', 'm_{4}', 'c_{9}'), ('m_{6}', 'c_{7}', 'm_{2}'), ('m_{6}', 'c_{7}', 'm_{3}'), ('m_{6}', 'c_{9}', 'm_{3}'), ('m_{6}', 'm_{4}', 'c_{7}'), ('m_{6}', 'm_{2}', 'c_{9}'), ('m_{6}', 'm_{6}', 'c_{9}'), ('m_{6}', 's_{2}', 'v_{11}'), ('m_{6}', 'c_{6}', 'c_{13}'), ('m_{6}', 'v_{8}', 's_{5}'), ('m_{6}', 's_{5}', 'v_{11}'), ('m_{6}', 'm_{7}', 'c_{13}'), ('m_{6}', 'c_{10}', 'c_{14}'), ('m_{6}', 'v_{8}', 's_{6}'), ('m_{6}', 'm_{7}', 'c_{14}'), ('m_{6}', 's_{6}', 'v_{11}'), ('m_{6}', 'm_{8}', 'c_{13}'), ('m_{6}', 's_{9}', 'v_{11}'), ('m_{6}', 'm_{11}', 'c_{13}'), ('m_{6}', 'v_{1}', 's_{9}'), ('m_{6}', 'm_{11}', 'c_{3}'), ('m_{6}', 'c_{3}', 'm_{13}'), ('m_{6}', 'c_{3}', 'm_{8}'), ('m_{6}', 'm_{13}', 'c_{13}'), ('m_{6}', 'c_{3}', 'c_{13}'), ('m_{6}', 'c_{7}', 'c_{14}'), ('m_{6}', 'm_{11}', 'c_{14}'), ('m_{6}', 'c_{7}', 'm_{8}'), ('m_{6}', 'm_{8}', 'c_{14}'), ('m_{6}', 'c_{7}', 'm_{13}'), ('m_{6}', 'm_{13}', 'c_{14}'), ('m_{6}', 'm_{11}', 'c_{7}'), ('m_{6}', 'v_{1}', 's_{10}'), ('m_{6}', 's_{10}', 'v_{11}'), ('m_{9}', 'c_{3}', 'm_{3}'), ('m_{9}', 'c_{5}', 'm_{3}'), ('m_{9}', 'v_{1}', 's_{1}'), ('m_{9}', 'm_{4}', 'c_{3}'), ('m_{9}', 'c_{3}', 'c_{5}'), ('m_{9}', 'm_{2}', 'c_{5}'), ('m_{9}', 'c_{3}', 'm_{6}'), ('m_{9}', 'm_{4}', 'c_{5}'), ('m_{9}', 'c_{3}', 'm_{2}'), ('m_{9}', 's_{1}', 'v_{11}'), ('m_{9}', 'm_{6}', 'c_{5}'), ('m_{9}', 'c_{7}', 'm_{6}'), ('m_{9}', 'v_{1}', 's_{2}'), ('m_{9}', 'c_{7}', 'c_{9}'), ('m_{9}', 'm_{4}', 'c_{9}'), ('m_{9}', 'c_{7}', 'm_{2}'), ('m_{9}', 'c_{7}', 'm_{3}'), ('m_{9}', 'c_{9}', 'm_{3}'), ('m_{9}', 'm_{4}', 'c_{7}'), ('m_{9}', 'm_{2}', 'c_{9}'), ('m_{9}', 'm_{6}', 'c_{9}'), ('m_{9}', 's_{2}', 'v_{11}'), ('m_{9}', 'c_{6}', 'c_{13}'), ('m_{9}', 'v_{8}', 's_{5}'), ('m_{9}', 's_{5}', 'v_{11}'), ('m_{9}', 'm_{7}', 'c_{13}'), ('m_{9}', 'c_{10}', 'c_{14}'), ('m_{9}', 'v_{8}', 's_{6}'), ('m_{9}', 'm_{7}', 'c_{14}'), ('m_{9}', 's_{6}', 'v_{11}'), ('m_{9}', 'm_{8}', 'c_{13}'), ('m_{9}', 's_{9}', 'v_{11}'), ('m_{9}', 'm_{11}', 'c_{13}'), ('m_{9}', 'v_{1}', 's_{9}'), ('m_{9}', 'm_{11}', 'c_{3}'), ('m_{9}', 'c_{3}', 'm_{13}'), ('m_{9}', 'c_{3}', 'm_{8}'), ('m_{9}', 'm_{13}', 'c_{13}'), ('m_{9}', 'c_{3}', 'c_{13}'), ('m_{9}', 'c_{7}', 'c_{14}'), ('m_{9}', 'm_{11}', 'c_{14}'), ('m_{9}', 'c_{7}', 'm_{8}'), ('m_{9}', 'm_{8}', 'c_{14}'), ('m_{9}', 'c_{7}', 'm_{13}'), ('m_{9}', 'm_{13}', 'c_{14}'), ('m_{9}', 'm_{11}', 'c_{7}'), ('m_{9}', 'v_{1}', 's_{10}'), ('m_{9}', 's_{10}', 'v_{11}'), ('m_{2}', 'c_{6}', 'm_{8}'), ('m_{2}', 'm_{9}', 'c_{6}'), ('m_{2}', 't_{5}', 'v_{10}'), ('m_{2}', 'm_{9}', 'c_{5}'), ('m_{2}', 'm_{6}', 'c_{6}'), ('m_{2}', 'm_{6}', 'c_{5}'), ('m_{2}', 'c_{6}', 'm_{7}'), ('m_{2}', 't_{6}', 'v_{10}'), ('m_{2}', 'c_{10}', 'm_{8}'), ('m_{2}', 'm_{9}', 'c_{9}'), ('m_{2}', 'm_{9}', 'c_{10}'), ('m_{2}', 't_{7}', 'v_{10}'), ('m_{2}', 'm_{6}', 'c_{9}'), ('m_{2}', 't_{8}', 'v_{10}'), ('m_{2}', 'm_{6}', 'c_{10}'), ('m_{2}', 'c_{10}', 'm_{7}'), ('m_{4}', 'c_{6}', 'm_{8}'), ('m_{4}', 'v_{6}', 't_{5}'), ('m_{4}', 'm_{9}', 'c_{6}'), ('m_{4}', 't_{5}', 'v_{10}'), ('m_{4}', 'm_{9}', 'c_{5}'), ('m_{4}', 'm_{6}', 'c_{6}'), ('m_{4}', 'm_{6}', 'c_{5}'), ('m_{4}', 'v_{6}', 't_{6}'), ('m_{4}', 'c_{6}', 'm_{7}'), ('m_{4}', 't_{6}', 'v_{10}'), ('m_{4}', 'c_{10}', 'm_{8}'), ('m_{4}', 'v_{6}', 't_{7}'), ('m_{4}', 'm_{9}', 'c_{9}'), ('m_{4}', 'm_{9}', 'c_{10}'), ('m_{4}', 't_{7}', 'v_{10}'), ('m_{4}', 'm_{6}', 'c_{9}'), ('m_{4}', 't_{8}', 'v_{10}'), ('m_{4}', 'm_{6}', 'c_{10}'), ('m_{4}', 'v_{6}', 't_{8}'), ('m_{4}', 'c_{10}', 'm_{7}'), ('c_{3}', 'c_{6}', 'm_{8}'), ('c_{3}', 'v_{6}', 't_{5}'), ('c_{3}', 'm_{9}', 'c_{6}'), ('c_{3}', 't_{5}', 'v_{10}'), ('c_{3}', 'm_{9}', 'c_{5}'), ('c_{3}', 'm_{6}', 'c_{6}'), ('c_{3}', 'm_{6}', 'c_{5}'), ('c_{3}', 'v_{6}', 't_{6}'), ('c_{3}', 'c_{6}', 'm_{7}'), ('c_{3}', 't_{6}', 'v_{10}'), ('c_{3}', 'c_{9}', 'm_{8}'), ('c_{3}', 'c_{10}', 'm_{8}'), ('c_{3}', 'v_{6}', 't_{7}'), ('c_{3}', 'm_{9}', 'c_{9}'), ('c_{3}', 'm_{9}', 'c_{10}'), ('c_{3}', 't_{7}', 'v_{10}'), ('c_{3}', 'm_{6}', 'c_{9}'), ('c_{3}', 't_{8}', 'v_{10}'), ('c_{3}', 'm_{6}', 'c_{10}'), ('c_{3}', 'v_{6}', 't_{8}'), ('c_{3}', 'c_{10}', 'm_{7}'), ('c_{3}', 'c_{9}', 'm_{7}'), ('c_{6}', 'm_{8}', 'm_{4}'), ('v_{6}', 't_{5}', 'm_{4}'), ('m_{9}', 'c_{6}', 'm_{4}'), ('t_{5}', 'v_{10}', 'm_{4}'), ('m_{9}', 'c_{5}', 'm_{4}'), ('c_{5}', 'm_{8}', 'm_{4}'), ('c_{6}', 'm_{8}', 'm_{7}'), ('v_{6}', 't_{5}', 'm_{7}'), ('t_{5}', 'v_{10}', 'm_{7}'), ('m_{9}', 'c_{5}', 'm_{7}'), ('c_{5}', 'm_{8}', 'm_{7}'), ('c_{6}', 'm_{8}', 'm_{13}'), ('v_{6}', 't_{5}', 'm_{13}'), ('m_{9}', 'c_{6}', 'm_{13}'), ('t_{5}', 'v_{10}', 'm_{13}'), ('m_{9}', 'c_{5}', 'm_{13}'), ('c_{5}', 'm_{8}', 'm_{13}'), ('c_{6}', 'm_{8}', 'c_{3}'), ('v_{6}', 't_{5}', 'c_{3}'), ('m_{9}', 'c_{6}', 'c_{3}'), ('t_{5}', 'v_{10}', 'c_{3}'), ('m_{9}', 'c_{5}', 'c_{3}'), ('c_{5}', 'm_{8}', 'c_{3}'), ('m_{6}', 'c_{6}', 'm_{4}'), ('c_{5}', 'm_{7}', 'm_{4}'), ('m_{6}', 'c_{5}', 'm_{4}'), ('v_{6}', 't_{6}', 'm_{4}'), ('c_{6}', 'm_{7}', 'm_{4}'), ('t_{6}', 'v_{10}', 'm_{4}'), ('c_{5}', 'm_{7}', 'm_{7}'), ('v_{6}', 't_{6}', 'm_{7}'), ('t_{6}', 'v_{10}', 'm_{7}'), ('m_{6}', 'c_{6}', 'm_{13}'), ('c_{5}', 'm_{7}', 'm_{13}'), ('m_{6}', 'c_{5}', 'm_{13}'), ('v_{6}', 't_{6}', 'm_{13}'), ('c_{6}', 'm_{7}', 'm_{13}'), ('t_{6}', 'v_{10}', 'm_{13}'), ('m_{6}', 'c_{6}', 'c_{3}'), ('c_{5}', 'm_{7}', 'c_{3}'), ('m_{6}', 'c_{5}', 'c_{3}'), ('v_{6}', 't_{6}', 'c_{3}'), ('c_{6}', 'm_{7}', 'c_{3}'), ('t_{6}', 'v_{10}', 'c_{3}'), ('c_{9}', 'm_{8}', 'm_{4}'), ('c_{10}', 'm_{8}', 'm_{4}'), ('v_{6}', 't_{7}', 'm_{4}'), ('m_{9}', 'c_{9}', 'm_{4}'), ('m_{9}', 'c_{10}', 'm_{4}'), ('t_{7}', 'v_{10}', 'm_{4}'), ('c_{9}', 'm_{8}', 'm_{7}'), ('c_{10}', 'm_{8}', 'm_{7}'), ('v_{6}', 't_{7}', 'm_{7}'), ('m_{9}', 'c_{9}', 'm_{7}'), ('t_{7}', 'v_{10}', 'm_{7}'), ('c_{9}', 'm_{8}', 'm_{13}'), ('c_{10}', 'm_{8}', 'm_{13}'), ('v_{6}', 't_{7}', 'm_{13}'), ('m_{9}', 'c_{9}', 'm_{13}'), ('m_{9}', 'c_{10}', 'm_{13}'), ('t_{7}', 'v_{10}', 'm_{13}'), ('c_{9}', 'm_{8}', 'c_{3}'), ('c_{10}', 'm_{8}', 'c_{3}'), ('v_{6}', 't_{7}', 'c_{3}'), ('m_{9}', 'c_{9}', 'c_{3}'), ('m_{9}', 'c_{10}', 'c_{3}'), ('t_{7}', 'v_{10}', 'c_{3}'), ('m_{6}', 'c_{9}', 'm_{4}'), ('t_{8}', 'v_{10}', 'm_{4}'), ('m_{6}', 'c_{10}', 'm_{4}'), ('v_{6}', 't_{8}', 'm_{4}'), ('c_{10}', 'm_{7}', 'm_{4}'), ('c_{9}', 'm_{7}', 'm_{4}'), ('t_{8}', 'v_{10}', 'm_{7}'), ('v_{6}', 't_{8}', 'm_{7}'), ('c_{9}', 'm_{7}', 'm_{7}'), ('m_{6}', 'c_{9}', 'm_{13}'), ('t_{8}', 'v_{10}', 'm_{13}'), ('m_{6}', 'c_{10}', 'm_{13}'), ('v_{6}', 't_{8}', 'm_{13}'), ('c_{10}', 'm_{7}', 'm_{13}'), ('c_{9}', 'm_{7}', 'm_{13}'), ('m_{6}', 'c_{9}', 'c_{3}'), ('t_{8}', 'v_{10}', 'c_{3}'), ('m_{6}', 'c_{10}', 'c_{3}'), ('v_{6}', 't_{8}', 'c_{3}'), ('c_{10}', 'm_{7}', 'c_{3}'), ('c_{9}', 'm_{7}', 'c_{3}'), ('c_{3}', 'm_{3}', 'm_{7}'), ('c_{5}', 'm_{3}', 'm_{7}'), ('m_{4}', 'c_{3}', 'm_{7}'), ('c_{3}', 'm_{6}', 'm_{7}'), ('c_{3}', 'm_{2}', 'm_{7}'), ('s_{1}', 'v_{11}', 'm_{7}'), ('c_{3}', 'm_{3}', 'm_{8}'), ('c_{5}', 'm_{3}', 'm_{8}'), ('m_{4}', 'c_{3}', 'm_{8}'), ('c_{3}', 'm_{6}', 'm_{8}'), ('c_{3}', 'm_{2}', 'm_{8}'), ('s_{1}', 'v_{11}', 'm_{8}'), ('m_{6}', 'c_{5}', 'm_{8}'), ('c_{7}', 'm_{6}', 'm_{7}'), ('c_{7}', 'c_{9}', 'm_{7}'), ('c_{7}', 'm_{2}', 'm_{7}'), ('c_{7}', 'm_{3}', 'm_{7}'), ('c_{9}', 'm_{3}', 'm_{7}'), ('m_{4}', 'c_{7}', 'm_{7}'), ('s_{2}', 'v_{11}', 'm_{7}'), ('c_{7}', 'm_{6}', 'm_{8}'), ('c_{7}', 'c_{9}', 'm_{8}'), ('c_{7}', 'm_{2}', 'm_{8}'), ('c_{7}', 'm_{3}', 'm_{8}'), ('c_{9}', 'm_{3}', 'm_{8}'), ('m_{4}', 'c_{7}', 'm_{8}'), ('m_{6}', 'c_{9}', 'm_{8}'), ('s_{2}', 'v_{11}', 'm_{8}'), ('c_{6}', 'c_{13}', 'm_{7}'), ('v_{8}', 's_{5}', 'm_{7}'), ('s_{5}', 'v_{11}', 'm_{7}'), ('m_{7}', 'c_{13}', 'm_{7}'), ('c_{6}', 'c_{13}', 'm_{8}'), ('v_{8}', 's_{5}', 'm_{8}'), ('c_{6}', 'm_{7}', 'm_{8}'), ('s_{5}', 'v_{11}', 'm_{8}'), ('m_{7}', 'c_{13}', 'm_{8}'), ('c_{10}', 'c_{14}', 'm_{7}'), ('v_{8}', 's_{6}', 'm_{7}'), ('m_{7}', 'c_{14}', 'm_{7}'), ('s_{6}', 'v_{11}', 'm_{7}'), ('c_{10}', 'm_{7}', 'm_{8}'), ('c_{10}', 'c_{14}', 'm_{8}'), ('v_{8}', 's_{6}', 'm_{8}'), ('m_{7}', 'c_{14}', 'm_{8}'), ('s_{6}', 'v_{11}', 'm_{8}'), ('m_{8}', 'c_{13}', 'm_{7}'), ('s_{9}', 'v_{11}', 'm_{7}'), ('m_{11}', 'c_{13}', 'm_{7}'), ('m_{11}', 'c_{3}', 'm_{7}'), ('c_{3}', 'm_{13}', 'm_{7}'), ('c_{3}', 'm_{8}', 'm_{7}'), ('m_{13}', 'c_{13}', 'm_{7}'), ('c_{3}', 'c_{13}', 'm_{7}'), ('m_{8}', 'c_{13}', 'm_{8}'), ('s_{9}', 'v_{11}', 'm_{8}'), ('m_{11}', 'c_{13}', 'm_{8}'), ('m_{11}', 'c_{3}', 'm_{8}'), ('c_{3}', 'm_{13}', 'm_{8}'), ('c_{3}', 'm_{8}', 'm_{8}'), ('m_{13}', 'c_{13}', 'm_{8}'), ('c_{3}', 'c_{13}', 'm_{8}'), ('c_{7}', 'c_{14}', 'm_{7}'), ('m_{11}', 'c_{14}', 'm_{7}'), ('c_{7}', 'm_{8}', 'm_{7}'), ('m_{8}', 'c_{14}', 'm_{7}'), ('c_{7}', 'm_{13}', 'm_{7}'), ('m_{13}', 'c_{14}', 'm_{7}'), ('m_{11}', 'c_{7}', 'm_{7}'), ('s_{10}', 'v_{11}', 'm_{7}'), ('c_{7}', 'c_{14}', 'm_{8}'), ('m_{11}', 'c_{14}', 'm_{8}'), ('c_{7}', 'm_{8}', 'm_{8}'), ('m_{8}', 'c_{14}', 'm_{8}'), ('c_{7}', 'm_{13}', 'm_{8}'), ('m_{13}', 'c_{14}', 'm_{8}'), ('m_{11}', 'c_{7}', 'm_{8}'), ('s_{10}', 'v_{11}', 'm_{8}'), ('m_{6}', 'v_{7}', 's_{1}'), ('v_{6}', 'm_{6}', 's_{1}'), ('m_{6}', 'v_{7}', 's_{2}'), ('v_{6}', 'm_{6}', 's_{2}'), ('m_{6}', 'v_{7}', 's_{5}'), ('v_{6}', 'm_{6}', 's_{5}'), ('m_{6}', 'v_{7}', 's_{6}'), ('v_{6}', 'm_{6}', 's_{6}'), ('m_{6}', 'v_{7}', 's_{9}'), ('v_{6}', 'm_{6}', 's_{9}'), ('m_{6}', 'v_{7}', 's_{10}'), ('v_{6}', 'm_{6}', 's_{10}'), ('m_{9}', 'v_{7}', 's_{1}'), ('v_{6}', 'm_{9}', 's_{1}'), ('m_{9}', 'v_{7}', 's_{2}'), ('v_{6}', 'm_{9}', 's_{2}'), ('m_{9}', 'v_{7}', 's_{5}'), ('v_{6}', 'm_{9}', 's_{5}'), ('m_{9}', 'v_{7}', 's_{6}'), ('v_{6}', 'm_{9}', 's_{6}'), ('m_{9}', 'v_{7}', 's_{9}'), ('v_{6}', 'm_{9}', 's_{9}'), ('m_{9}', 'v_{7}', 's_{10}'), ('v_{6}', 'm_{9}', 's_{10}'), ('v_{3}', 'm_{2}', 't_{5}'), ('v_{3}', 'm_{2}', 't_{6}'), ('v_{3}', 'm_{2}', 't_{7}'), ('v_{3}', 'm_{2}', 't_{8}'), ('m_{4}', 'v_{2}', 't_{5}'), ('m_{4}', 'v_{2}', 't_{6}'), ('m_{4}', 'v_{2}', 't_{7}'), ('m_{4}', 'v_{2}', 't_{8}'), ('v_{2}', 'c_{3}', 't_{5}'), ('c_{3}', 'v_{3}', 't_{5}'), ('v_{2}', 'c_{3}', 't_{6}'), ('c_{3}', 'v_{3}', 't_{6}'), ('v_{2}', 'c_{3}', 't_{7}'), ('c_{3}', 'v_{3}', 't_{7}'), ('v_{2}', 'c_{3}', 't_{8}'), ('c_{3}', 'v_{3}', 't_{8}'), ('m_{9}', 'c_{5}', 'c_{6}'), ('m_{9}', 'c_{5}', 'm_{8}'), ('c_{5}', 'c_{6}', 'm_{8}'), ('m_{9}', 'c_{6}', 'm_{8}'), ('c_{5}', 'c_{6}', 'm_{7}'), ('m_{6}', 'c_{6}', 'm_{7}'), ('m_{6}', 'c_{5}', 'm_{7}'), ('m_{6}', 'c_{5}', 'c_{6}'), ('m_{9}', 'c_{9}', 'c_{10}'), ('m_{9}', 'c_{9}', 'm_{8}'), ('c_{9}', 'c_{10}', 'm_{8}'), ('m_{9}', 'c_{10}', 'm_{8}'), ('c_{9}', 'c_{10}', 'm_{7}'), ('m_{6}', 'c_{10}', 'm_{7}'), ('m_{6}', 'c_{9}', 'm_{7}'), ('m_{6}', 'c_{9}', 'c_{10}')], 'h2_0': [('v_{1}', 's_{5}', 'm_{10}'), ('v_{1}', 's_{5}', 'm_{5}'), ('v_{1}', 's_{6}', 'm_{10}'), ('v_{1}', 's_{6}', 'm_{5}'), ('v_{1}', 's_{7}', 'm_{10}'), ('v_{1}', 's_{7}', 'm_{5}'), ('v_{1}', 's_{8}', 'm_{10}'), ('v_{1}', 's_{8}', 'm_{5}'), ('m_{11}', 's_{5}', 'v_{1}'), ('m_{11}', 's_{6}', 'v_{1}'), ('m_{11}', 's_{7}', 'v_{1}'), ('m_{11}', 's_{8}', 'v_{1}'), ('m_{11}', 's_{9}', 'v_{1}'), ('m_{11}', 's_{10}', 'v_{1}'), ('m_{4}', 's_{5}', 'v_{1}'), ('m_{4}', 's_{6}', 'v_{1}'), ('m_{4}', 's_{7}', 'v_{1}'), ('m_{4}', 's_{8}', 'v_{1}'), ('m_{4}', 's_{9}', 'v_{1}'), ('m_{4}', 's_{10}', 'v_{1}'), ('s_{5}', 'm_{10}', 'v_{1}'), ('s_{5}', 'm_{5}', 'v_{1}'), ('s_{6}', 'm_{10}', 'v_{1}'), ('s_{6}', 'm_{5}', 'v_{1}'), ('s_{7}', 'm_{10}', 'v_{1}'), ('s_{7}', 'm_{5}', 'v_{1}'), ('s_{8}', 'm_{10}', 'v_{1}'), ('s_{8}', 'm_{5}', 'v_{1}'), ('s_{9}', 'm_{10}', 'v_{1}'), ('s_{9}', 'm_{5}', 'v_{1}'), ('s_{10}', 'm_{10}', 'v_{1}'), ('s_{10}', 'm_{5}', 'v_{1}'), ('t_{1}', 'm_{1}', 'v_{1}'), ('t_{1}', 'm_{3}', 'v_{1}'), ('t_{1}', 'm_{12}', 'v_{1}'), ('t_{2}', 'm_{1}', 'v_{1}'), ('t_{2}', 'm_{3}', 'v_{1}'), ('t_{2}', 'm_{12}', 'v_{1}'), ('t_{3}', 'm_{1}', 'v_{1}'), ('t_{3}', 'm_{3}', 'v_{1}'), ('t_{3}', 'm_{12}', 'v_{1}'), ('t_{4}', 'm_{1}', 'v_{1}'), ('t_{4}', 'm_{3}', 'v_{1}'), ('t_{4}', 'm_{12}', 'v_{1}'), ('m_{11}', 'c_{6}', 'c_{13}'), ('m_{11}', 'v_{8}', 's_{5}'), ('m_{11}', 'c_{6}', 'm_{7}'), ('m_{11}', 's_{5}', 'v_{11}'), ('m_{11}', 'm_{7}', 'c_{13}'), ('m_{11}', 'c_{10}', 'm_{7}'), ('m_{11}', 'c_{10}', 'c_{14}'), ('m_{11}', 'v_{8}', 's_{6}'), ('m_{11}', 'm_{7}', 'c_{14}'), ('m_{11}', 's_{6}', 'v_{11}'), ('m_{11}', 'm_{10}', 'c_{6}'), ('m_{11}', 'v_{3}', 's_{7}'), ('m_{11}', 'm_{12}', 'c_{6}'), ('m_{11}', 's_{7}', 'v_{9}'), ('m_{11}', 'c_{4}', 'c_{6}'), ('m_{11}', 'v_{3}', 's_{8}'), ('m_{11}', 'm_{10}', 'c_{10}'), ('m_{11}', 'm_{12}', 'c_{10}'), ('m_{11}', 'c_{8}', 'c_{10}'), ('m_{11}', 's_{8}', 'v_{9}'), ('m_{11}', 'm_{8}', 'c_{13}'), ('m_{11}', 's_{9}', 'v_{11}'), ('m_{11}', 'm_{11}', 'c_{13}'), ('m_{11}', 'v_{1}', 's_{9}'), ('m_{11}', 'm_{11}', 'c_{3}'), ('m_{11}', 'c_{3}', 'm_{13}'), ('m_{11}', 'c_{3}', 'm_{8}'), ('m_{11}', 'm_{13}', 'c_{13}'), ('m_{11}', 'c_{3}', 'c_{13}'), ('m_{11}', 'c_{7}', 'c_{14}'), ('m_{11}', 'm_{11}', 'c_{14}'), ('m_{11}', 'c_{7}', 'm_{8}'), ('m_{11}', 'm_{8}', 'c_{14}'), ('m_{11}', 'c_{7}', 'm_{13}'), ('m_{11}', 'm_{13}', 'c_{14}'), ('m_{11}', 'm_{11}', 'c_{7}'), ('m_{11}', 'v_{1}', 's_{10}'), ('m_{11}', 's_{10}', 'v_{11}'), ('m_{4}', 'c_{6}', 'c_{13}'), ('m_{4}', 'v_{8}', 's_{5}'), ('m_{4}', 'c_{6}', 'm_{7}'), ('m_{4}', 's_{5}', 'v_{11}'), ('m_{4}', 'm_{7}', 'c_{13}'), ('m_{4}', 'c_{10}', 'm_{7}'), ('m_{4}', 'c_{10}', 'c_{14}'), ('m_{4}', 'v_{8}', 's_{6}'), ('m_{4}', 'm_{7}', 'c_{14}'), ('m_{4}', 's_{6}', 'v_{11}'), ('m_{4}', 'm_{10}', 'c_{6}'), ('m_{4}', 'v_{3}', 's_{7}'), ('m_{4}', 'm_{12}', 'c_{6}'), ('m_{4}', 's_{7}', 'v_{9}'), ('m_{4}', 'c_{4}', 'c_{6}'), ('m_{4}', 'c_{4}', 'm_{10}'), ('m_{4}', 'v_{3}', 's_{8}'), ('m_{4}', 'c_{8}', 'm_{10}'), ('m_{4}', 'm_{10}', 'c_{10}'), ('m_{4}', 'm_{12}', 'c_{10}'), ('m_{4}', 'c_{8}', 'c_{10}'), ('m_{4}', 's_{8}', 'v_{9}'), ('m_{4}', 'm_{8}', 'c_{13}'), ('m_{4}', 's_{9}', 'v_{11}'), ('m_{4}', 'm_{11}', 'c_{13}'), ('m_{4}', 'v_{1}', 's_{9}'), ('m_{4}', 'm_{11}', 'c_{3}'), ('m_{4}', 'c_{3}', 'm_{13}'), ('m_{4}', 'c_{3}', 'm_{8}'), ('m_{4}', 'm_{13}', 'c_{13}'), ('m_{4}', 'c_{3}', 'c_{13}'), ('m_{4}', 'c_{7}', 'c_{14}'), ('m_{4}', 'm_{11}', 'c_{14}'), ('m_{4}', 'c_{7}', 'm_{8}'), ('m_{4}', 'm_{8}', 'c_{14}'), ('m_{4}', 'c_{7}', 'm_{13}'), ('m_{4}', 'm_{13}', 'c_{14}'), ('m_{4}', 'm_{11}', 'c_{7}'), ('m_{4}', 'v_{1}', 's_{10}'), ('m_{4}', 's_{10}', 'v_{11}'), ('s_{5}', 'v_{4}', 'm_{10}'), ('s_{5}', 'm_{10}', 'v_{5}'), ('s_{5}', 'm_{5}', 'v_{5}'), ('s_{5}', 'v_{4}', 'm_{5}'), ('s_{6}', 'v_{4}', 'm_{10}'), ('s_{6}', 'm_{10}', 'v_{5}'), ('s_{6}', 'm_{5}', 'v_{5}'), ('s_{6}', 'v_{4}', 'm_{5}'), ('s_{7}', 'v_{4}', 'm_{10}'), ('s_{7}', 'm_{10}', 'v_{5}'), ('s_{7}', 'm_{5}', 'v_{5}'), ('s_{7}', 'v_{4}', 'm_{5}'), ('s_{8}', 'v_{4}', 'm_{10}'), ('s_{8}', 'm_{10}', 'v_{5}'), ('s_{8}', 'm_{5}', 'v_{5}'), ('s_{8}', 'v_{4}', 'm_{5}'), ('s_{9}', 'v_{4}', 'm_{10}'), ('s_{9}', 'm_{10}', 'v_{5}'), ('s_{9}', 'm_{5}', 'v_{5}'), ('s_{9}', 'v_{4}', 'm_{5}'), ('s_{10}', 'v_{4}', 'm_{10}'), ('s_{10}', 'm_{10}', 'v_{5}'), ('s_{10}', 'm_{5}', 'v_{5}'), ('s_{10}', 'v_{4}', 'm_{5}'), ('t_{1}', 'm_{1}', 'v_{11}'), ('t_{1}', 'v_{1}', 'm_{1}'), ('t_{1}', 'm_{3}', 'v_{11}'), ('t_{1}', 'v_{8}', 'm_{3}'), ('t_{1}', 'm_{12}', 'v_{8}'), ('t_{2}', 'm_{1}', 'v_{11}'), ('t_{2}', 'v_{1}', 'm_{1}'), ('t_{2}', 'm_{3}', 'v_{11}'), ('t_{2}', 'v_{8}', 'm_{3}'), ('t_{2}', 'm_{12}', 'v_{8}'), ('t_{3}', 'm_{1}', 'v_{11}'), ('t_{3}', 'v_{1}', 'm_{1}'), ('t_{3}', 'm_{3}', 'v_{11}'), ('t_{3}', 'v_{8}', 'm_{3}'), ('t_{3}', 'm_{12}', 'v_{8}'), ('t_{4}', 'm_{1}', 'v_{11}'), ('t_{4}', 'v_{1}', 'm_{1}'), ('t_{4}', 'm_{3}', 'v_{11}'), ('t_{4}', 'v_{8}', 'm_{3}'), ('t_{4}', 'm_{12}', 'v_{8}'), ('m_{11}', 'v_{2}', 's_{5}'), ('m_{11}', 'v_{2}', 's_{6}'), ('m_{11}', 'v_{2}', 's_{7}'), ('m_{11}', 'v_{2}', 's_{8}'), ('m_{11}', 'v_{2}', 's_{9}'), ('m_{11}', 'v_{2}', 's_{10}'), ('m_{4}', 'v_{2}', 's_{5}'), ('m_{4}', 'v_{2}', 's_{6}'), ('m_{4}', 'v_{2}', 's_{7}'), ('m_{4}', 'v_{2}', 's_{8}'), ('m_{4}', 'v_{2}', 's_{9}'), ('m_{4}', 'v_{2}', 's_{10}'), ('c_{6}', 'c_{13}', 'm_{10}'), ('v_{8}', 's_{5}', 'm_{10}'), ('c_{6}', 'm_{7}', 'm_{10}'), ('s_{5}', 'v_{11}', 'm_{10}'), ('m_{7}', 'c_{13}', 'm_{10}'), ('c_{6}', 'c_{13}', 'm_{5}'), ('v_{8}', 's_{5}', 'm_{5}'), ('c_{6}', 'm_{7}', 'm_{5}'), ('s_{5}', 'v_{11}', 'm_{5}'), ('m_{7}', 'c_{13}', 'm_{5}'), ('c_{10}', 'm_{7}', 'm_{10}'), ('c_{10}', 'c_{14}', 'm_{10}'), ('v_{8}', 's_{6}', 'm_{10}'), ('m_{7}', 'c_{14}', 'm_{10}'), ('s_{6}', 'v_{11}', 'm_{10}'), ('c_{10}', 'm_{7}', 'm_{5}'), ('c_{10}', 'c_{14}', 'm_{5}'), ('v_{8}', 's_{6}', 'm_{5}'), ('m_{7}', 'c_{14}', 'm_{5}'), ('s_{6}', 'v_{11}', 'm_{5}'), ('m_{10}', 'c_{6}', 'm_{10}'), ('v_{3}', 's_{7}', 'm_{10}'), ('c_{4}', 'm_{12}', 'm_{10}'), ('m_{12}', 'c_{6}', 'm_{10}'), ('s_{7}', 'v_{9}', 'm_{10}'), ('c_{4}', 'c_{6}', 'm_{10}'), ('c_{4}', 'm_{10}', 'm_{10}'), ('m_{10}', 'c_{6}', 'm_{5}'), ('v_{3}', 's_{7}', 'm_{5}'), ('c_{4}', 'm_{12}', 'm_{5}'), ('m_{12}', 'c_{6}', 'm_{5}'), ('s_{7}', 'v_{9}', 'm_{5}'), ('c_{4}', 'c_{6}', 'm_{5}'), ('c_{4}', 'm_{10}', 'm_{5}'), ('v_{3}', 's_{8}', 'm_{10}'), ('c_{8}', 'm_{10}', 'm_{10}'), ('m_{10}', 'c_{10}', 'm_{10}'), ('c_{8}', 'm_{12}', 'm_{10}'), ('m_{12}', 'c_{10}', 'm_{10}'), ('c_{8}', 'c_{10}', 'm_{10}'), ('s_{8}', 'v_{9}', 'm_{10}'), ('v_{3}', 's_{8}', 'm_{5}'), ('c_{8}', 'm_{10}', 'm_{5}'), ('m_{10}', 'c_{10}', 'm_{5}'), ('c_{8}', 'm_{12}', 'm_{5}'), ('m_{12}', 'c_{10}', 'm_{5}'), ('c_{8}', 'c_{10}', 'm_{5}'), ('s_{8}', 'v_{9}', 'm_{5}'), ('m_{8}', 'c_{13}', 'm_{10}'), ('s_{9}', 'v_{11}', 'm_{10}'), ('m_{11}', 'c_{13}', 'm_{10}'), ('c_{3}', 'm_{13}', 'm_{10}'), ('c_{3}', 'm_{8}', 'm_{10}'), ('m_{13}', 'c_{13}', 'm_{10}'), ('c_{3}', 'c_{13}', 'm_{10}'), ('m_{8}', 'c_{13}', 'm_{5}'), ('s_{9}', 'v_{11}', 'm_{5}'), ('m_{11}', 'c_{13}', 'm_{5}'), ('m_{11}', 'c_{3}', 'm_{5}'), ('c_{3}', 'm_{13}', 'm_{5}'), ('c_{3}', 'm_{8}', 'm_{5}'), ('m_{13}', 'c_{13}', 'm_{5}'), ('c_{3}', 'c_{13}', 'm_{5}'), ('c_{7}', 'c_{14}', 'm_{10}'), ('m_{11}', 'c_{14}', 'm_{10}'), ('c_{7}', 'm_{8}', 'm_{10}'), ('m_{8}', 'c_{14}', 'm_{10}'), ('c_{7}', 'm_{13}', 'm_{10}'), ('m_{13}', 'c_{14}', 'm_{10}'), ('s_{10}', 'v_{11}', 'm_{10}'), ('c_{7}', 'c_{14}', 'm_{5}'), ('m_{11}', 'c_{14}', 'm_{5}'), ('c_{7}', 'm_{8}', 'm_{5}'), ('m_{8}', 'c_{14}', 'm_{5}'), ('c_{7}', 'm_{13}', 'm_{5}'), ('m_{13}', 'c_{14}', 'm_{5}'), ('m_{11}', 'c_{7}', 'm_{5}'), ('s_{10}', 'v_{11}', 'm_{5}'), ('c_{3}', 'm_{10}', 'm_{1}'), ('m_{11}', 'c_{3}', 'm_{1}'), ('m_{11}', 'c_{4}', 'm_{1}'), ('c_{4}', 'm_{10}', 'm_{1}'), ('t_{1}', 'v_{5}', 'm_{1}'), ('c_{3}', 'm_{10}', 'm_{3}'), ('m_{11}', 'c_{3}', 'm_{3}'), ('m_{11}', 'c_{4}', 'm_{3}'), ('c_{4}', 'm_{10}', 'm_{3}'), ('t_{1}', 'v_{5}', 'm_{3}'), ('c_{3}', 'm_{10}', 'm_{12}'), ('m_{11}', 'c_{3}', 'm_{12}'), ('c_{4}', 'm_{10}', 'm_{12}'), ('m_{4}', 'c_{4}', 'm_{1}'), ('m_{4}', 'c_{3}', 'm_{1}'), ('t_{2}', 'v_{5}', 'm_{1}'), ('c_{3}', 'm_{5}', 'm_{1}'), ('c_{4}', 'm_{5}', 'm_{1}'), ('m_{4}', 'c_{4}', 'm_{3}'), ('m_{4}', 'c_{3}', 'm_{3}'), ('t_{2}', 'v_{5}', 'm_{3}'), ('c_{3}', 'm_{5}', 'm_{3}'), ('c_{4}', 'm_{5}', 'm_{3}'), ('m_{4}', 'c_{3}', 'm_{12}'), ('c_{3}', 'm_{5}', 'm_{12}'), ('c_{4}', 'm_{5}', 'm_{12}'), ('m_{11}', 'c_{8}', 'm_{1}'), ('c_{8}', 'm_{10}', 'm_{1}'), ('t_{3}', 'v_{5}', 'm_{1}'), ('m_{11}', 'c_{7}', 'm_{1}'), ('c_{7}', 'm_{10}', 'm_{1}'), ('m_{11}', 'c_{8}', 'm_{3}'), ('c_{8}', 'm_{10}', 'm_{3}'), ('t_{3}', 'v_{5}', 'm_{3}'), ('m_{11}', 'c_{7}', 'm_{3}'), ('c_{7}', 'm_{10}', 'm_{3}'), ('c_{8}', 'm_{10}', 'm_{12}'), ('m_{11}', 'c_{7}', 'm_{12}'), ('c_{7}', 'm_{10}', 'm_{12}'), ('m_{4}', 'c_{7}', 'm_{1}'), ('t_{4}', 'v_{5}', 'm_{1}'), ('c_{8}', 'm_{5}', 'm_{1}'), ('c_{7}', 'm_{5}', 'm_{1}'), ('m_{4}', 'c_{8}', 'm_{1}'), ('m_{4}', 'c_{7}', 'm_{3}'), ('t_{4}', 'v_{5}', 'm_{3}'), ('c_{8}', 'm_{5}', 'm_{3}'), ('c_{7}', 'm_{5}', 'm_{3}'), ('m_{4}', 'c_{8}', 'm_{3}'), ('m_{4}', 'c_{7}', 'm_{12}'), ('c_{8}', 'm_{5}', 'm_{12}'), ('c_{7}', 'm_{5}', 'm_{12}'), ('c_{3}', 'c_{4}', 'm_{10}'), ('m_{11}', 'c_{3}', 'c_{4}'), ('c_{3}', 'c_{4}', 'm_{5}'), ('m_{4}', 'c_{4}', 'm_{5}'), ('m_{4}', 'c_{3}', 'm_{5}'), ('m_{4}', 'c_{3}', 'c_{4}'), ('c_{7}', 'c_{8}', 'm_{10}'), ('m_{11}', 'c_{7}', 'c_{8}'), ('c_{7}', 'c_{8}', 'm_{5}'), ('m_{4}', 'c_{8}', 'm_{5}'), ('m_{4}', 'c_{7}', 'm_{5}'), ('m_{4}', 'c_{7}', 'c_{8}')], 'h1_0': [('m_{11}', 'm_{4}', 'v_{1}'), ('m_{4}', 'm_{4}', 'v_{1}'), ('m_{11}', 'v_{1}', 'm_{4}'), ('m_{11}', 'm_{4}', 'v_{2}'), ('m_{4}', 'v_{1}', 'm_{4}'), ('m_{4}', 'm_{4}', 'v_{2}'), ('m_{11}', 'v_{2}', 'm_{4}'), ('m_{4}', 'v_{2}', 'm_{4}')], 'h1_1': [('v_{1}', 'c_{3}', 'm_{4}'), ('v_{1}', 'c_{3}', 'c_{3}'), ('v_{1}', 'c_{7}', 'm_{4}'), ('v_{1}', 'c_{7}', 'c_{3}'), ('m_{4}', 'c_{3}', 'v_{1}'), ('m_{4}', 'c_{7}', 'v_{1}'), ('c_{3}', 'm_{4}', 'v_{1}'), ('c_{3}', 'c_{3}', 'v_{1}'), ('c_{7}', 'm_{4}', 'v_{1}'), ('c_{7}', 'c_{3}', 'v_{1}'), ('m_{4}', 'c_{3}', 'v_{3}'), ('m_{4}', 'c_{7}', 'v_{3}'), ('c_{3}', 'v_{1}', 'm_{4}'), ('c_{3}', 'm_{4}', 'v_{2}'), ('c_{3}', 'v_{2}', 'c_{3}'), ('c_{3}', 'c_{3}', 'v_{3}'), ('c_{7}', 'v_{1}', 'm_{4}'), ('c_{7}', 'm_{4}', 'v_{2}'), ('c_{7}', 'v_{2}', 'c_{3}'), ('c_{7}', 'c_{3}', 'v_{3}'), ('v_{2}', 'c_{3}', 'm_{4}'), ('c_{3}', 'v_{3}', 'm_{4}'), ('v_{2}', 'c_{3}', 'c_{3}'), ('c_{3}', 'v_{3}', 'c_{3}'), ('c_{7}', 'v_{3}', 'm_{4}'), ('v_{2}', 'c_{7}', 'm_{4}'), ('c_{7}', 'v_{3}', 'c_{3}'), ('v_{2}', 'c_{7}', 'c_{3}')], 'h1_2': [('v_{1}', 'm_{2}', 'm_{6}'), ('v_{1}', 'm_{2}', 'm_{9}'), ('v_{1}', 'c_{3}', 'm_{6}'), ('v_{1}', 'c_{3}', 'm_{9}'), ('v_{1}', 'm_{6}', 'm_{2}'), ('v_{1}', 'm_{6}', 'm_{4}'), ('v_{1}', 'm_{6}', 'm_{6}'), ('v_{1}', 'm_{6}', 'c_{3}'), ('v_{1}', 'm_{9}', 'm_{2}'), ('v_{1}', 'm_{9}', 'm_{4}'), ('v_{1}', 'm_{9}', 'm_{6}'), ('v_{1}', 'm_{9}', 'c_{3}'), ('m_{2}', 'm_{6}', 'v_{1}'), ('m_{2}', 'm_{9}', 'v_{1}'), ('m_{4}', 'm_{6}', 'v_{1}'), ('m_{4}', 'm_{9}', 'v_{1}'), ('c_{3}', 'm_{6}', 'v_{1}'), ('c_{3}', 'm_{9}', 'v_{1}'), ('m_{6}', 'm_{2}', 'v_{1}'), ('m_{6}', 'm_{4}', 'v_{1}'), ('m_{6}', 'm_{6}', 'v_{1}'), ('m_{6}', 'c_{3}', 'v_{1}'), ('m_{9}', 'm_{2}', 'v_{1}'), ('m_{9}', 'm_{4}', 'v_{1}'), ('m_{9}', 'm_{6}', 'v_{1}'), ('m_{9}', 'c_{3}', 'v_{1}'), ('m_{2}', 'm_{6}', 'v_{7}'), ('m_{2}', 'm_{9}', 'v_{7}'), ('m_{4}', 'm_{6}', 'v_{7}'), ('m_{4}', 'v_{6}', 'm_{6}'), ('m_{4}', 'm_{9}', 'v_{7}'), ('m_{4}', 'v_{6}', 'm_{9}'), ('c_{3}', 'm_{6}', 'v_{7}'), ('c_{3}', 'v_{6}', 'm_{6}'), ('c_{3}', 'm_{9}', 'v_{7}'), ('c_{3}', 'v_{6}', 'm_{9}'), ('m_{6}', 'v_{3}', 'm_{2}'), ('m_{6}', 'm_{2}', 'v_{6}'), ('m_{6}', 'v_{1}', 'm_{4}'), ('m_{6}', 'm_{4}', 'v_{2}'), ('m_{6}', 'm_{6}', 'v_{7}'), ('m_{6}', 'v_{6}', 'm_{6}'), ('m_{6}', 'v_{2}', 'c_{3}'), ('m_{6}', 'c_{3}', 'v_{3}'), ('m_{9}', 'v_{3}', 'm_{2}'), ('m_{9}', 'm_{2}', 'v_{6}'), ('m_{9}', 'v_{1}', 'm_{4}'), ('m_{9}', 'm_{4}', 'v_{2}'), ('m_{9}', 'm_{6}', 'v_{7}'), ('m_{9}', 'v_{6}', 'm_{6}'), ('m_{9}', 'v_{2}', 'c_{3}'), ('m_{9}', 'c_{3}', 'v_{3}'), ('v_{3}', 'm_{2}', 'm_{6}'), ('v_{3}', 'm_{2}', 'm_{9}'), ('m_{4}', 'v_{2}', 'm_{6}'), ('m_{4}', 'v_{2}', 'm_{9}'), ('v_{2}', 'c_{3}', 'm_{6}'), ('c_{3}', 'v_{3}', 'm_{6}'), ('v_{2}', 'c_{3}', 'm_{9}'), ('c_{3}', 'v_{3}', 'm_{9}'), ('m_{6}', 'v_{7}', 'm_{2}'), ('v_{6}', 'm_{6}', 'm_{2}'), ('m_{6}', 'v_{7}', 'm_{4}'), ('v_{6}', 'm_{6}', 'm_{4}'), ('m_{6}', 'v_{7}', 'm_{6}'), ('v_{6}', 'm_{6}', 'm_{6}'), ('m_{6}', 'v_{7}', 'c_{3}'), ('v_{6}', 'm_{6}', 'c_{3}'), ('m_{9}', 'v_{7}', 'm_{2}'), ('v_{6}', 'm_{9}', 'm_{2}'), ('m_{9}', 'v_{7}', 'm_{4}'), ('v_{6}', 'm_{9}', 'm_{4}'), ('m_{9}', 'v_{7}', 'm_{6}'), ('v_{6}', 'm_{9}', 'm_{6}'), ('m_{9}', 'v_{7}', 'c_{3}'), ('v_{6}', 'm_{9}', 'c_{3}')]}
    BR_phi_1_factored = {k: factorize_cycles(v, BR_C) for k, v in BR_phi_1.iteritems()}
    print "\nfactored(phi_1) =", BR_phi_1_factored

    # test to ensure that all elements from original are still in factored
    BR_phi_1_expanded = expand_map_all(BR_phi_1_factored)
    print "\nfactored(phi_1) = phi_1? ",
    print not any(add_maps_mod_2(BR_phi_1, BR_phi_1_expanded).values())
    if any(add_maps_mod_2(BR_phi_1, BR_phi_1_expanded).values()):
        print add_maps_mod_2(BR_phi_1, BR_phi_1_expanded)

    BR_delta3 = {k: [tuple(map(f_BR, list(t))) for t in tuples] for k, tuples in BR_phi_1_factored.iteritems()}
    BR_delta3 = chain_map_mod(expand_map_all({k: [tp for tp in vs if all(tp)] for k, vs in BR_delta3.iteritems()}))
    print "\nDelta_3 =", BR_delta3

    found_cycles = True
    while found_cycles:
        found_cycles = False
        # (g x g x g) Delta3
        BR_gxgxg_delta3 = {k: [(g_l, g_m, g_r) for l, m, r in v for g_l in BR_g[l] for g_m in BR_g[m] for g_r in BR_g[r]] for k, v in BR_delta3.iteritems()}
        print "\n(g x g x g) Delta_3 =", BR_gxgxg_delta3

        BR_nabla_g3 = add_maps_mod_2(BR_gxgxg_delta3, BR_phi_1)
        print "\nNabla(g^3) =", BR_nabla_g3
        BR_nabla_g3_factored = {k: factorize_cycles(v, BR_C) for k, v in BR_nabla_g3.iteritems()}
        residual_delta3 = {k: [tuple(map(f_BR, list(t))) for t in tuples] for k, tuples in BR_nabla_g3_factored.iteritems()}
        residual_delta3 = chain_map_mod(expand_map_all(residual_delta3))

        if any(residual_delta3.values()):
            found_cycles = True
            print "Found residual Delta3!"
            BR_delta3 = add_maps_mod_2(BR_delta3, residual_delta3)
            print "\nDelta_3 =", BR_delta3

    BR_nabla_nabla_g3 = {k: [(l, m, r) for (l, m, r) in derivative(v, BR_C) if l and m and r] for k, v in BR_nabla_g3.iteritems() if v}
    BR_nabla_nabla_g3 = chain_map_mod(expand_map_all(BR_nabla_nabla_g3))
    print "\nNabla(Nabla(g^3)) =", BR_nabla_nabla_g3

    print "\nNabla(g^3) factored =", BR_nabla_g3_factored
    print "\nsums nabla(g^3) factored =", {k: len(tuples) for k, tuples in BR_nabla_g3_factored.iteritems()}

    BR_nabla_g3_boundary_groups = {k: group_boundary_chains(tuples, BR_C) for k, tuples in BR_nabla_g3_factored.iteritems()}
    BR_nabla_g3_boundary_groups = {k: [factorize(group, BR_C) for group in groups] for k, groups in BR_nabla_g3_boundary_groups.iteritems()}
    print "\nNabla(g^3) boundary grouped =", BR_nabla_g3_boundary_groups

    # check what the image of f map looks like
    # we might need this for additional grouping
    # for k, groups in BR_nabla_g3_boundary_groups.iteritems():
    #     print "\n", k, " :\n["
    #     for group in groups:
    #         for chain in group:
    #             print "\t", tuple(map(f_BR, list(chain)))
    #         print "]\n["

    # verify that all groups are cycles
    BR_nabla_nabla_g3_boundary_groups = {}

    for k, groups in BR_nabla_g3_boundary_groups.iteritems():
        groups_expanded = []
        for tuples in groups:
            d_tuples = [(l, m, r) for t in tuples for (l, m, r) in derivative(t, BR_C) if l and m and r]
            d_tuples_expanded = [exp_dt for dt in d_tuples for exp_dt in expand_tuple_list(dt)]
            groups_expanded.append(list_mod(d_tuples_expanded))
        BR_nabla_nabla_g3_boundary_groups[k] = groups_expanded
    print "\nNumber of groups:", {k: len(groups) for k, groups in BR_nabla_nabla_g3_boundary_groups.iteritems()}
    print "\nAll groups are cycles? =", not any([any(vs) for vs in BR_nabla_nabla_g3_boundary_groups.values()])

    BR_g3_grouped = {k: [] for k in BR_nabla_g3_boundary_groups.keys()}
    BR_nabla_g3_grouped_remainder = {k: [] for k in BR_nabla_g3_boundary_groups.keys()}
    for k, groups in BR_nabla_g3_boundary_groups.iteritems():
        for group in reversed(groups):
            anti_group = group_integrate(group, BR_C)
            if anti_group is None:
                BR_nabla_g3_grouped_remainder[k].append(group)
            else:
                print "\n\nanti(", group, ") = "
                print "\t", factorize_cycles(anti_group, BR_C)
                BR_g3_grouped[k].append(anti_group)

    print "\ng^3 (partial)=", BR_g3_grouped
    print "\nnabla(g^3) remainder:", BR_nabla_g3_grouped_remainder
    exit()

    BR_g3 = {k: integrate_BR(vs) for k, vs in BR_nabla_g3.iteritems()}
    BR_g3 = chain_map_mod(expand_map_all(BR_g3))
    print "\ng^3 =", BR_g3

    BR_nabla_g3_computed = {k: [(l, m, r) for (l, m, r) in derivative(v, BR_C) if l and m and r] for k, v in BR_g3.iteritems() if v}
    BR_nabla_g3_computed = chain_map_mod(expand_map_all(BR_nabla_g3_computed))
    print "\nNabla(g^3) (computed) =", BR_nabla_g3_computed

    print "\nNabla(g^3) + (g x g x g) Delta_3 + phi_1 =", add_maps_mod_2(BR_nabla_g3_computed, BR_nabla_g3)


    print "\nderivative({a_{1}: [(m_{1},m_{1},m_{1})]}) =", derivative({'a_{1}': [('m_{1}','m_{1}','m_{1}')]}, BR_C)

    print "\nfacet_to_cells(v_{1}) =", facet_to_cells('v_{1}', BR_C)




if __name__ == '__main__':
    main()
