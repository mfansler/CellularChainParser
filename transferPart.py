# Standard imports
from subprocess import check_output, CalledProcessError
from os import remove as rm

from re import sub, compile
from argparse import ArgumentParser

from itertools import product, combinations
from collections import Counter

import numpy
import scipy.sparse as sp

# Local imports
import CellChainParse
from Coalgebra import Coalgebra
from factorize import factorize_recursive as factorize

__author__ = 'mfansler'
temp_mat = "~transfer-temp.mat"

# chars
DELTA   = u"\u0394"
PARTIAL = u"\u2202"
NABLA   = u"\u2207"
OTIMES  = u"\u2297"
CHAINPARTIAL = "(1" + OTIMES + PARTIAL + " + " + PARTIAL + OTIMES + "1)"
THETA = u"\u03b8"
PHI = u"\u03c6"


def format_cells(cells):
    return sub(',', '_', sub(r'[{}]', '', str(cells)))


def format_tuple(t):
    if type(t) is tuple:
        return u" \u2297 ".join(list(t))
    else:
        return unicode(t)


def format_sum(obj):
    if obj is None:
        return "0"
    elif type(obj) is dict:
        single = [format_tuple(k) for k, v in obj.items() if v == 1]
        multiple = [u"{}*({})".format(v, format_tuple(k)) for k, v in obj.items() if v > 1]
        return u" + ".join(single + multiple)
    elif type(obj) is list:
        return u"(" + u" + ".join([format_tuple(o) for o in obj]) + ")"
    else:
        return obj


def format_morphism(m):
    return u"\n\t+ ".join([u"{}{}_{{{}}}".format('(' + format_morphism(v) + ')' if type(v) is dict else format_sum(v), PARTIAL, k) for k, v in m.items()])


def compare_incidences(x, y):
    return x[1] - y[1] if x[1] != y[1] else x[0] - y[0]


def tensor(*groups):

    def mult(xs):
        return reduce(lambda a, b: a*b, xs, 1) + 0

    maxes = [max(g.keys()) for g in groups]
    tensor_groups = {i: [] for i in range(sum(maxes) + 1)}

    for combin in product(*[range(m+1) for m in maxes]):
        tensor_groups[sum(combin)] += product(*[groups[i][combin[i]] for i in range(len(groups))])

    return tensor_groups


def list_mod(ls, modulus=2):
    return [s for s, num in Counter(ls).items() if num % modulus]


def add_maps_mod_2(a, b):

    res = a
    for k, vals in b.items():
        if k not in res:
            res[k] = vals
        else:
            for v in vals:
                if v in res[k]:
                    res[k] = [u for u in res[k] if u != v]
                else:
                    res[k].append(v)

    return res


def chain_coproduct(chain, coproduct, simplify=True):

    ps = [p for cell in chain for p, num in coproduct[cell].items() if num % 2]
    return [el for el, num in Counter(ps).items() if num % 2] if simplify else ps


def get_vector_in_basis(el, basis):

    # el = [{}, {}]
    # basis = [ {'h0_0': ['a','b','c',..,'z']}, {'h0_1': ['b', 'c', ...]}, ..., {}]
    return [1 if k in el and v in el[k] else 0 for b in basis for k, v in b.items()]


hom_dim_re = compile('h(\d*)_')


def hom_dim(h_element):
    return int(hom_dim_re.match(h_element).group(1))


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


# generates all 0-n combinations of elements in the list xs
def all_combinations(xs):
    for i in range(len(xs) + 1):
        for c in combinations(xs, i):
            yield c
    #return (c for i in range(len(xs)+1) for c in combinations(xs, i))


# generates the function f: C -> H
# @param C Coalgebra to map from
# @param g map from H (== H*(C)) to class representatives in C
#
# returns function f(x)
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

    # extract just the (partial) inverse
    inv_mat = rref_mat.tocsc()[:, n:].tocsr()

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

    return f


argparser = ArgumentParser(description="Computes induced coproduct on homology")
argparser.add_argument('file', type=file, help="LaTeX file to be parsed")
args = None

try:
    args = argparser.parse_args()
except Exception as e:
    print e.strerror
    argparser.print_help()
    raise SystemExit

# load file
data = args.file.read()
args.file.close()

# parse file contents
result = CellChainParse.parse(data)

if not result:
    raise SystemExit

# construct coalgebra
C = Coalgebra(result["groups"], result["differentials"], result["coproducts"])


"""
COMPUTE HOMOLOGY
"""

differential = {n: C.incidence_matrix(n, sparse=False) for n in range(1, C.topDimension() + 1)}

# create temporary file for CHomP
scratch = open(temp_mat, 'w+')

for n, entries in differential.iteritems():
    print >> scratch, n - 1
    incidences = [(l, r, v % 2) for (l, r), v in entries.iteritems()]
    for entry in ["{} {} {}".format(l, r, v) for (l, r, v) in sorted(incidences, cmp=compare_incidences)]:
        print >> scratch, entry

scratch.close()

try:
    chomp_results = check_output(["chomp-matrix", temp_mat, "-g"])
    print chomp_results
except CalledProcessError as e:
    print e.returncode
    print e.output
    print e.output
finally:
    rm(temp_mat)  # clean up

lines = chomp_results.splitlines()

dims = [int(k) for k in compile('\d+').findall(lines[0])]


H_gens = {}
offset = 9 + len(dims)
for n, k in enumerate(dims):
    H_gens[n] = [[C.groups[n][int(j)] for j in compile('\[(\d+)\]').findall(lines[offset + i])] for i in range(k)]
    offset += k + 1

# Manually entering results from SageMath for basis for homology
#H_gens[0] = [['v_{1}']]
#H_gens[1] = [['m_{11}', 'm_{4}'], ['c_{3}', 'c_{7}'], ['m_{6}', 'm_{9}']]
#H_gens[2] = [['t_{1}', 't_{2}', 't_{3}', 't_{4}'], ['t_{5}', 't_{6}', 't_{7}', 't_{8}']]
#H_gens[3] = []

H = {dim: ["h{}_{}".format(dim, i) for i, gen in enumerate(gens)] for dim, gens in H_gens.items()}
print "H = H*(C) = ", H

# Define g
g = {"h{}_{}".format(dim, i): gen for dim, gens in H_gens.items() for i, gen in enumerate(gens) if gen}
print
print "g = ", format_morphism(g)

# generate f: C -> H
f = compute_f(C, g)

# define Delta g
Delta_g = {k: chain_coproduct(v, C.coproduct) for k, v in g.items()}
print
print DELTA + u"g =", format_morphism(Delta_g)

print
print DELTA + u"g (unsimplified) =", {k: chain_coproduct(v, C.coproduct, simplify=False) for k, v in g.items()}


# CxC = tensor(C.groups, C.groups)
#
# dCxC = {}
# for k, vs in CxC.items():
#     dCxC[k] = {}
#     for (l, r) in vs:
#         dLeft = [(l_i, r) for l_i in C.differential[l]] if l in C.differential else []
#         dRight = [(l, r_i) for r_i in C.differential[r]] if r in C.differential else []
#         if dLeft + dRight:
#             dCxC[k][(l, r)] = dLeft + dRight

factored_delta_g = {k: factorize(v) for k, v in Delta_g.items()}
print
print DELTA + u"g (factored) =", factored_delta_g

delta2 = {k: [tuple(map(f, list(t))) for t in tuples] for k, tuples in factored_delta_g.items()}
print delta2
exit()
g2 = {k: [] for k in g.keys()}

#print "H->dCxC = ", H_to_dCxC_1
# for k, v in g.items():
#     dim = int(k[1])+1
#     img = chain_coproduct(v, C.coproduct)
#     g2[k] = []
#     for chain, bd in dCxC[dim].items():
#         if all([cell in img for cell in bd]):
#             g2[k].append(chain)
#             for cell in bd:
#                 img.remove(cell)
#     delta2[k] = img
#
# print
# print DELTA + u"_2 =", format_morphism(delta2)
# print
# print u"g^2 =", format_morphism(g2)
# print
#print 'H = ', H


# basis for vector space
def H_to_CxC_0():
    return ({h: cxc} for dim, hs in H.items() for h in hs for cxc in CxC[dim])

# express Delta g in that basis
delta_g_vec = get_vector_in_basis(Delta_g, H_to_CxC_0())


# generate all possible components of Delta_2, that is Hom_0(H, HxH)
HxH = tensor(H, H)


def H_to_HxH_0():
    return ({h: hxh} for dim, hs in H.items() for h in hs for hxh in HxH[dim])


# convert all possible Delta_2 components to H -> HxH -> CxC
# note that we are keeping the original maps associated with them so they don't get lost
def g_x_g_H_to_HxH_0():
    return ((hs, {h: [(l, r) for l in g[h_l] for r in g[h_r]] for h, (h_l, h_r) in hs.items()}) for hs in H_to_HxH_0())


# express the Delta2 components in the Hom_0(H, CxC) vector space
def g_x_g_H_to_HxH_0_vecs():
    return ((hs, get_vector_in_basis(h_to_cxcs, H_to_CxC_0())) for (hs, h_to_cxcs) in g_x_g_H_to_HxH_0())


# generate all components in the Hom_0(H, dCxC) space
def H_to_dCxC_1():
    return (({h: cxc}, {h: dcxc}) for dim, hs in H.items() for h in hs for cxc, dcxc in dCxC[dim+1].items() if dcxc)


def H_to_dCxC_1_vecs():
    return ((hs, get_vector_in_basis(h_to_cxcs, H_to_CxC_0())) for (hs, h_to_cxcs) in H_to_dCxC_1())

#
X_img = numpy.array([vec for (_, vec) in g_x_g_H_to_HxH_0_vecs()], dtype=numpy.int8).transpose()
X_ker = numpy.array([vec for (_, vec) in H_to_dCxC_1_vecs()], dtype=numpy.int8).transpose()
y = numpy.array([delta_g_vec], dtype=numpy.int8).transpose()

img_size = X_img.shape[1]
input_matrix = numpy.append(numpy.append(X_img, X_ker, axis=1), y, axis=1)
#print input_matrix
sols_mat = row_reduce_mod2(input_matrix, -1)
numpy.set_printoptions(threshold=numpy.nan)


for i in [i for i in numpy.nonzero(sols_mat[:, -1])[0]]:
    # get leftmost non-zero column
    j = numpy.nonzero(sols_mat[i, :])[0][0]
    #print j
    if j < img_size:
        for h, hxh in list(g_x_g_H_to_HxH_0_vecs())[j][0].items():
            delta2[h] = delta2[h] + [hxh] if h in delta2 else [hxh]
        #print g_x_g_H_to_HxH_0_vecs[i]
    else:
        for h, cxc in list(H_to_dCxC_1())[j - img_size][0].items():
            g2[h] = g2[h] + [cxc] if h in g2 else [cxc]
        #print H_to_dCxC_1[i - len(g_x_g_H_to_HxH_0_vecs)]

print
print DELTA + u"_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta2.items()})

# g^2
print
print u"g^2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g2.items() if v})

# (g x g) Delta
gxgDelta = {k: [(g_l, g_r) for l, r in v for g_l in g[l] for g_r in g[r]] for k, v in delta2.items()}
print
print u"(g " + OTIMES + " g)" + DELTA + "_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in gxgDelta.items()})

# VERIFY: DELTA g = (g x g) DELTA_2 + NABLA g^2

# NABLA g^2
nabla_g2 = {}
for k, vs in g2.items():
    nabla_g2[k] = []
    for (l, r) in vs:
        dLeft = [(l_i, r) for l_i in C.differential[l]] if l in C.differential else []
        dRight = [(l, r_i) for r_i in C.differential[r]] if r in C.differential else []
        if dLeft + dRight:
            nabla_g2[k] += dLeft + dRight

print
print NABLA + u" g^2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in nabla_g2.items() if v})

# (g x g) DELTA_2 + NABLA g^2
sum_Delta_g = add_maps_mod_2(gxgDelta, nabla_g2)
print
print u"(g " + OTIMES + " g)" + DELTA + "_2 + " + NABLA + " g^2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in sum_Delta_g.items()})

# Delta g = (g x g) DELTA_2 + NABLA g^2
print
print DELTA + " g = (g " + OTIMES + " g)" + DELTA + "_2 + " + NABLA + " g^2 : ",
print all([vs == [] for vs in add_maps_mod_2(sum_Delta_g, Delta_g).values()])

#--------------------------------------------#



# (1 x Delta) g^2
id_x_Delta_g2 = {k: [(l,) + r_cp for (l, r) in v for r_cp in C.coproduct[r].keys()] for k, v in g2.items()}
print
print u"(1 " + OTIMES + " " + DELTA + ") g^2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in id_x_Delta_g2.items() if v})

# (Delta x 1) g^2
Delta_x_id_g2 = {k: [l_cp + (r,) for (l, r) in v for l_cp in C.coproduct[l].keys()] for k, v in g2.items()}
print
print u"(" + DELTA + " " + OTIMES + " 1) g^2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in Delta_x_id_g2.items() if v})

# (g x g^2) Delta_2
g_x_g2_Delta2 = {k: [(l_cp,) + t for l, r in v for t in g2[r] for l_cp in g[l]] for k, v in delta2.items()}
print
print u"( g " + OTIMES + " g^2 ) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in g_x_g2_Delta2.items() if v})

# (g^2 x g) Delta_2
g2_x_g_Delta2 = {k: [t + (r_cp,) for l, r in v for t in g2[l] for r_cp in g[r]] for k, v in delta2.items()}
print
print u"( g^2 " + OTIMES + " g ) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in g2_x_g_Delta2.items() if v})

# (1 x Delta_2) Delta_2
id_x_Delta2_Delta2 = {k: [(l,) + r_cp for (l, r) in v for r_cp in delta2[r]] for k, v in delta2.items()}
print
print u"(1 " + OTIMES + " " + DELTA + "_2) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in id_x_Delta2_Delta2.items() if v})

# (Delta_2 x 1) Delta_2
Delta2_x_id_Delta2 = {k: [l_cp + (r,) for (l, r) in v for l_cp in delta2[l]] for k, v in delta2.items()}
print
print u"(" + DELTA + "_2 " + OTIMES + " 1) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in Delta2_x_id_Delta2.items() if v})

# z_1 = (1 x Delta_2 + Delta_2 x 1) Delta_2
z_1 = add_maps_mod_2(id_x_Delta2_Delta2, Delta2_x_id_Delta2)
print
print u"z_1 = (1 " + OTIMES + " " + DELTA + "_2 + " + DELTA + "_2 " + OTIMES + " 1) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in z_1.items() if v})

# phi_1 = (1 x Delta + Delta x 1) g^2 + (g x g^2 + g^2 x g) Delta_2
phi_1 = reduce(add_maps_mod_2, [g_x_g2_Delta2, g2_x_g_Delta2, id_x_Delta_g2, Delta_x_id_g2], {})
print
print PHI + u"_1 = (g " + OTIMES + " g^2 + g^2 " + OTIMES + " g) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in phi_1.items() if v})

#print "DEBUG: generating CxCxC"
CxCxC = tensor(C.groups, C.groups, C.groups)
#print "DEBUG: CxCxC size ", {k: len(vs) for k, vs in CxCxC.items()}

#print "DEBUG: generating dCxCxC"
dCxCxC = {}
for k, vs in CxCxC.items():
    dCxCxC[k] = {}
    for (l, m, r) in vs:
        dLeft = [(l_i, m, r) for l_i in C.differential[l]] if l in C.differential else []
        dMiddle = [(l, m_i, r) for m_i in C.differential[m]] if m in C.differential else []
        dRight = [(l, m, r_i) for r_i in C.differential[r]] if r in C.differential else []
        if dLeft + dMiddle + dRight:
            dCxCxC[k][(l, m, r)] = dLeft + dMiddle + dRight


# basis for vector space
def H_to_CxCxC_1():
    return ({h: cxcxc} for dim, hs in H.items() for h in hs for cxcxc in CxCxC[dim + 1])


#print "DEBUG: generating phi_1 vector"
phi_1_vec = get_vector_in_basis(phi_1, H_to_CxCxC_1())

#print "DEBUG: generating HxHxH"
HxHxH = tensor(H, H, H)


def H_to_HxHxH_1():
    #print "\tDEBUG: enter H_to_HxHxH_0 generator"
    return ({h: hxhxh} for dim, hs in H.items() for h in hs for hxhxh in HxHxH[dim + 1])


def gxgxg_H_to_HxHxH_1():
    #print "\tDEBUG: enter gxgxg_H_to_HxHxH_0 generator"
    return ((hs, {h: [(l, m, r) for l in g[h_l] for m in g[h_m] for r in g[h_r]]
                  for h, (h_l, h_m, h_r) in hs.items()}) for hs in H_to_HxHxH_1())


def gxgxg_H_to_HxHxH_1_vecs():
    #print "\tDEBUG: enter gxgxg_H_to_HxHxH_0_vecs generator"
    return ((hs, get_vector_in_basis(h_to_cxcxcs, H_to_CxCxC_1())) for (hs, h_to_cxcxcs) in gxgxg_H_to_HxHxH_1())


# generate all components in the Hom_0(H, dCxCxC) space
def H_to_dCxCxC_1():
    #print "\tDEBUG: enter H_to_dCxCxC_1 generator"
    return (({h: cxcxc}, {h: dcxcxc}) for dim, hs in H.items() for h in hs for cxcxc, dcxcxc in dCxCxC[dim+2].items() if dcxcxc)


def H_to_dCxCxC_1_vecs():
    #print "\tDEBUG: enter H_to_dCxCxC_1_vecs generator"
    return ((hs, get_vector_in_basis(h_to_cxcxcs, H_to_CxCxC_1())) for (hs, h_to_cxcxcs) in H_to_dCxCxC_1())

#print "DEBUG: Generating Image (h --> CxCxC) matrix"
X_img = numpy.array([vec for (_, vec) in gxgxg_H_to_HxHxH_1_vecs()]).transpose()
img_size = X_img.shape[1]

#print "DEBUG: Generating Kernel (h --> CxCxC) matrix"
X_ker = numpy.array([vec for (_, vec) in H_to_dCxCxC_1_vecs()]).transpose()

#print "DEBUG: Generating Phi (h --> CxCxC) vector"
y = numpy.array([phi_1_vec]).transpose()

#print "DEBUG: appending matrices"
input_matrix = numpy.append(numpy.append(X_img, X_ker, axis=1), y, axis=1)

#print "DEBUG: Row reducing matrix"
sols_mat = row_reduce_mod2(input_matrix, -1)

delta3 = {k: [] for k in g.keys()}
g3 = {k: [] for k in g.keys()}

#print "DEBUG: Extracting solution"
for i in [i for i in numpy.nonzero(sols_mat[:, -1])[0]]:
    # get leftmost non-zero column
    j = numpy.nonzero(sols_mat[i, :])[0][0]
    if j < img_size:
        for h, hxhxh in list(gxgxg_H_to_HxHxH_1_vecs())[j][0].items():
            delta3[h] = delta3[h] + [hxhxh]
    else:
        for h, cxcxc in list(H_to_dCxCxC_1())[j - img_size][0].items():
            g3[h] = g3[h] + [cxcxc]


print
print DELTA + u"_3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta3.items() if v})

# g^3
print
print u"g^3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g3.items() if v})

# Clean up
del X_img, X_ker, y, phi_1_vec, z_1, HxHxH, CxCxC, dCxCxC, input_matrix, sols_mat

