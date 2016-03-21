# Standard imports
from subprocess import check_output, CalledProcessError
from os import remove as rm

from re import sub
from re import compile
from argparse import ArgumentParser

from itertools import product
from collections import Counter

import numpy
from scipy import linalg

# Local imports
import CellChainParse
from Coalgebra import Coalgebra

__author__ = 'mfansler'
temp_mat = "~transfer-temp.mat"

# chars
DELTA   = u"\u0394"
PARTIAL = u"\u2202"
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
    tmp = numpy.copy(A[r1, :])
    A[r1, :] = A[r2, :]
    A[r2, :] = tmp


def row_reduce_mod2(A, augment=-1):

    if A.ndim != 2:
        print A.ndim
        raise Exception("require two dimensional matrix input")

    A = numpy.fmod(A, 2)
    rank = 0
    for i in range(A.shape[1] + augment):

        nzs = numpy.nonzero(A[rank:, i])[0]
        if len(nzs) > 0:

            row_swap(A, rank, rank + nzs[0])

            for nz in nzs[1:]:
                A[rank + nz, :] = numpy.fmod(A[rank + nz, :] + A[rank, :], 2)
            if i > 0:
                for nz in numpy.nonzero(A[:rank, i])[0]:
                    A[nz, :] = numpy.fmod(A[nz, :] + A[rank, :], 2)
            rank += 1
    return A


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
    #print "debug: ", C.groups[n]
    H_gens[n] = [[C.groups[n][int(j)] for j in compile('\[(\d+)\]').findall(lines[offset + i])] for i in range(k)]
    #print "debug: ", lines[offset], k
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

# IMPORTANT: g_inv can no longer be defined
# Define g inverse
# g_inv = {v: k for k, v in g.items()}

alpha = {}
beta = {}

# K2 = Delta
alpha[THETA+'2'] = C.coproduct

# J1 -> g

beta['f1'] = g

# BEGIN DEBUG
# for k, vs in g.items():
#     print vs, " => ", '(' + str([res for v in vs for res in C.coproduct[v].keys()]) + ')'
#    print vs, " => ", chain_coproduct(vs, C.coproduct)
# END DEBUG

# J2: theta2 f1 -> Delta g
beta[THETA+'2f1'] = {}
for k, vs in g.items():
    beta[THETA+'2f1'][k] = '(' + format_sum(chain_coproduct(vs, C.coproduct)) + ')'

# define Delta g
Delta_g = {k: chain_coproduct(v, C.coproduct) for k, v in g.items()}
print
print DELTA + u"g =", format_morphism(Delta_g)

CxC = tensor(C.groups, C.groups)

dCxC = {}
for k, vs in CxC.items():
    dCxC[k] = {}
    for (l, r) in vs:
        dLeft = [(l_i, r) for l_i in C.differential[l]] if l in C.differential else []
        dRight = [(l, r_i) for r_i in C.differential[r]] if r in C.differential else []
        if dLeft + dRight:
            dCxC[k][(l, r)] = dLeft + dRight


delta2 = {}
g2 = {}

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
H_to_CxC_0 = [{h: cxc} for dim, hs in H.items() for h in hs for cxc in CxC[dim]]
#print "size[H->CxC] = ", len(H_to_CxC_0)

# express Delta g in that basis
#print Delta_g
delta_g_vec = get_vector_in_basis(Delta_g, H_to_CxC_0)
#print len(delta_g_vec)

# generate all possible components of Delta_2, that is Hom_0(H, HxH)
HxH = tensor(H, H)
H_to_HxH_0 = [{h: hxh} for dim, hs in H.items() for h in hs for hxh in HxH[dim]]

# convert all possible Delta_2 components to H -> HxH -> CxC
# note that we are keeping the original maps associated with them so they don't get lost
g_x_g_H_to_HxH_0 = [(hs, {h: [(l,r) for l in g[h_l] for r in g[h_r]] for h, (h_l, h_r) in hs.items()}) for hs in H_to_HxH_0]
#print "size[H->HxH] = ", len(H_to_HxH_0)
#print "size[(gxg)(H->HxH)] = ", len(g_x_g_H_to_HxH_0)
#print g_x_g_H_to_HxH_0

# express the Delta2 components in the Hom_0(H, CxC) vector space
g_x_g_H_to_HxH_0_vecs = [(hs, get_vector_in_basis(h_to_cxcs, H_to_CxC_0)) for (hs, h_to_cxcs) in g_x_g_H_to_HxH_0]
#print g_x_g_H_to_HxH_0_vecs

# generate all components in the Hom_0(H, dCxC) space
H_to_dCxC_1 = [({h: cxc}, {h: dcxc}) for dim, hs in H.items() for h in hs for cxc, dcxc in dCxC[dim+1].items() if dcxc]
#print "size[(H->dCxC)] = ", len(H_to_dCxC_1)
#print H_to_dCxC_1
H_to_dCxC_1_vecs = [(hs, get_vector_in_basis(h_to_cxcs, H_to_CxC_0)) for (hs, h_to_cxcs) in H_to_dCxC_1]

#
X_img = numpy.array([vec for (_, vec) in g_x_g_H_to_HxH_0_vecs]).transpose()
X_ker = numpy.array([vec for (_, vec) in H_to_dCxC_1_vecs]).transpose()
y = numpy.array([delta_g_vec]).transpose()
#print X_img.shape, X_ker.shape, y.shape
#print numpy.append(numpy.append(X_img, X_ker, axis=1), y, axis=1).shape
#print sum(numpy.all(X == 0, axis=0)), sum(numpy.all(y == 1, axis=0))
#print numpy.append(X_img, X_ker, axis=1).shape
#print numpy.linalg.matrix_rank(numpy.append(X_img, X_ker, axis=1))
input_matrix = numpy.append(numpy.append(X_img, X_ker, axis=1), y, axis=1)
#print input_matrix
sols_mat = row_reduce_mod2(input_matrix, -1)
numpy.set_printoptions(threshold=numpy.nan)

#print sols_mat


vs = [i for i in numpy.nonzero(sols_mat[:, -1])[0]]
for i in vs:
    # get leftmost non-zero column
    j = numpy.nonzero(sols_mat[i, :])[0][0]
    #print j
    if j < len(g_x_g_H_to_HxH_0_vecs):
        for h, hxh in g_x_g_H_to_HxH_0_vecs[j][0].items():
            delta2[h] = delta2[h] + [hxh] if h in delta2 else [hxh]
        #print g_x_g_H_to_HxH_0_vecs[i]
    else:
        for h, cxc in H_to_dCxC_1[j - len(g_x_g_H_to_HxH_0_vecs)][0].items():
            g2[h] = g2[h] + [cxc] if h in g2 else [cxc]
        #print H_to_dCxC_1[i - len(g_x_g_H_to_HxH_0_vecs)]

print
print DELTA + u"_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta2.items()})

# g^2
print
print u"g^2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g2.items() if v})
