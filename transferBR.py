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

# Debugging
#numpy.set_printoptions(threshold=numpy.nan)


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
# scratch = open(temp_mat, 'w+')
#
# for n, entries in differential.iteritems():
#     print >> scratch, n - 1
#     incidences = [(l, r, v % 2) for (l, r), v in entries.iteritems()]
#     for entry in ["{} {} {}".format(l, r, v) for (l, r, v) in sorted(incidences, cmp=compare_incidences)]:
#         print >> scratch, entry
#
# scratch.close()
#
# try:
#     chomp_results = check_output(["chomp-matrix", temp_mat, "-g"])
#     print chomp_results
# except CalledProcessError as e:
#     print e.returncode
#     print e.output
#     print e.output
# finally:
#     rm(temp_mat)  # clean up
#
# lines = chomp_results.splitlines()
#
# dims = [int(k) for k in compile('\d+').findall(lines[0])]


H_gens = {}
# offset = 9 + len(dims)
# for n, k in enumerate(dims):
#     #print "debug: ", C.groups[n]
#     H[n] = [[C.groups[n][int(j)] for j in compile('\[(\d+)\]').findall(lines[offset + i])] for i in range(k)]
#     #print "debug: ", lines[offset], k
#     offset += k + 1

# Manually entering results from SageMath for basis for homology
H_gens[0] = [['v_{1}']]
H_gens[1] = [['m_{11}', 'm_{4}'], ['c_{3}', 'c_{7}'], ['m_{6}', 'm_{9}']]
H_gens[2] = [['t_{1}', 't_{2}', 't_{3}', 't_{4}'], ['t_{5}', 't_{6}', 't_{7}', 't_{8}']]
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

# print "dCxC[0] = ", dCxC[0]
# print "dCxC[1] = ",
# for k, v in dCxC[1].items():
#     print "\t", v
# print
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
g_x_g_H_to_HxH_0 = [(hs, {h: [(l, r) for l in g[h_l] for r in g[h_r]] for h, (h_l, h_r) in hs.items()}) for hs in H_to_HxH_0]
#print "size[H->HxH] = ", len(H_to_HxH_0)
#print "size[(gxg)(H->HxH)] = ", len(g_x_g_H_to_HxH_0)
# for thing in g_x_g_H_to_HxH_0:
#     print thing

# express the Delta2 components in the Hom_0(H, CxC) vector space
g_x_g_H_to_HxH_0_vecs = [(hs, get_vector_in_basis(h_to_cxcs, H_to_CxC_0)) for (hs, h_to_cxcs) in g_x_g_H_to_HxH_0]
#print [(hs, sum(vec)) for (hs, vec) in g_x_g_H_to_HxH_0_vecs]

# generate all components in the Hom_0(H, dCxC) space
H_to_dCxC_1 = [({h: cxc}, {h: dcxc}) for dim, hs in H.items() for h in hs for cxc, dcxc in dCxC[dim+1].items() if dcxc]
#print "size[(H->dCxC)] = ", len(H_to_dCxC_1)

H_to_dCxC_1_vecs = [(hs, get_vector_in_basis(h_to_cxcs, H_to_CxC_0)) for (hs, h_to_cxcs) in H_to_dCxC_1]
# print H_to_dCxC_1[:10]

X_img = numpy.array([vec for (_, vec) in g_x_g_H_to_HxH_0_vecs]).transpose()
X_ker = numpy.array([vec for (_, vec) in H_to_dCxC_1_vecs]).transpose()
# rank(ker) = 5014
y = numpy.array([delta_g_vec]).transpose()

#rref_ker = row_reduce_mod2(X_ker, 0)
#print "nonzero rows = ", sum(numpy.any(rref_ker[:,:5425] != 0, axis=1))

#print X_img.shape, X_ker.shape, y.shape
#print numpy.append(numpy.append(X_img, X_ker, axis=1), y, axis=1).shape
#print sum(numpy.all(X == 0, axis=0)), sum(numpy.all(y == 1, axis=0))
#print numpy.append(X_img, X_ker, axis=1).shape
#print numpy.linalg.matrix_rank(numpy.append(X_img, X_ker, axis=1))
input_matrix = numpy.append(numpy.append(X_img, X_ker, axis=1), y, axis=1)
sols_mat = row_reduce_mod2(input_matrix, -1)

#print sols_mat

delta2 = {}
g2 = {}

vs = [i for i in numpy.nonzero(sols_mat[:, -1])[0] if sols_mat[i, i]]
for i in vs:
    # get leftmost non-zero column
    j = numpy.nonzero(sols_mat[i, :])[0][0]
    if j < len(g_x_g_H_to_HxH_0_vecs):
        for h, hxh in g_x_g_H_to_HxH_0_vecs[j][0].items():
            delta2[h] = delta2[h] + [hxh] if h in delta2 else [hxh]
        #print g_x_g_H_to_HxH_0_vecs[j][0]
    else:
        for h, cxc in H_to_dCxC_1[j - len(g_x_g_H_to_HxH_0_vecs)][0].items():
            g2[h] = g2[h] + [cxc] if h in g2 else [cxc]
        #print H_to_dCxC_1[j - len(g_x_g_H_to_HxH_0_vecs)]

print
print DELTA + u"_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta2.items()})

# g^2
print
print u"g^2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g2.items() if v})

# (g x g) Delta
gxgDelta = {k: [(g_l, g_r) for l, r in v for g_l in g[l] for g_r in g[r]] for k, v in delta2.items()}
#print gxgDelta
print
print u"(g " + OTIMES + " g)" + DELTA + "_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in gxgDelta.items()})

# (1 x Delta) g^2
id_x_Delta_g2 = {k: [(l,) + r_cp for (l, r) in v for r_cp in C.coproduct[r].keys()] for k, v in g2.items()}
print
print u"(1 " + OTIMES + " " + DELTA + ") g^2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in id_x_Delta_g2.items() if v})

# (Delta x 1) g^2
Delta_x_id_g2 = {k: [l_cp + (r,) for (l, r) in v for l_cp in C.coproduct[l].keys()] for k, v in g2.items()}
print
print u"(" + DELTA + " " + OTIMES + " 1) g^2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in Delta_x_id_g2.items() if v})

# (g x g^2) Delta_2
g_x_g2_Delta2 = {k: [(g[l],) + t for l, r in v for t in g2[r]] for k, v in delta2.items()}
print
print u"( g " + OTIMES + " g^2 ) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in g_x_g2_Delta2.items() if v})

# (g^2 x g) Delta_2
g2_x_g_Delta2 = {k: [t + (g[r],) for l, r in v for t in g2[l]] for k, v in delta2.items()}
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
phi_1 = add_maps_mod_2(add_maps_mod_2(g_x_g2_Delta2, g2_x_g_Delta2), add_maps_mod_2(id_x_Delta_g2, Delta_x_id_g2))
print
print PHI + u"_1 = (g " + OTIMES + " g^2 + g^2 " + OTIMES + " g) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in phi_1.items() if v})


CxCxC = tensor(C.groups, C.groups, C.groups)
# print CxCxC # debug

dCxCxC = {}
for k, vs in CxCxC.items():
    dCxCxC[k] = {}
    for (l, m, r) in vs:
        dLeft = [(l_i, m, r) for l_i in C.differential[l]] if l in C.differential else []
        dMiddle = [(l, m_i, r) for m_i in C.differential[m]] if m in C.differential else []
        dRight = [(l, m, r_i) for r_i in C.differential[r]] if r in C.differential else []
        if dLeft + dMiddle + dRight:
            dCxCxC[k][(l, m, r)] = dLeft + dMiddle + dRight

#print CxCxC
#print dCxCxC # debug

#print [g[k] for k in phi_1.keys() if phi_1[k]]


delta3 = {}
g3 = {} # all boundary found in image will become part of g3
for k, v in g.items():
    dim = int(k[1])+2

    img = phi_1[k]
    g3[k] = []
    for chain, bd in dCxCxC[dim].items():
        if all([cell in img for cell in bd]):
            g3[k].append(chain)
            for cell in bd:
                img.remove(cell)
    delta3[k] = img

delta3 = {k: [(g_inv[l], g_inv[m], g_inv[r]) for (l, m, r) in v] for k, v in delta3.items() if v}

print
print DELTA + u"_3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta3.items()})

# g^3
print
print u"g^3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g3.items() if v})

#####################
# Facets of J_4
#####################

# (1 x 1 x Delta) g^3
id_x_id_x_Delta_g3 = {k: [(l, m) + r_cp for (l, m, r) in v for r_cp in C.coproduct[r].keys()] for k, v in g3.items()}
print
print u"(1 " + OTIMES + " 1 " + OTIMES + " " + DELTA + ") g^3 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in id_x_id_x_Delta_g3.items() if v})

# (1 x Delta x 1) g^3
id_x_Delta_x_id_g3 = {k: [(l, ) + m_cp + (r, ) for (l, m, r) in v for m_cp in C.coproduct[m].keys()] for k, v in g3.items()}
print
print u"(1 " + OTIMES + " " + DELTA + " " + OTIMES + " 1) g^3 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in id_x_Delta_x_id_g3.items() if v})

# (Delta x 1 x 1) g^3
Delta_x_id_x_id_g3 = {k: [l_cp + (m, r) for (l, m, r) in v for l_cp in C.coproduct[l].keys()] for k, v in g3.items()}
print
print u"(" + DELTA + " " + OTIMES + " 1 " + OTIMES + " 1) g^3 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in Delta_x_id_x_id_g3.items() if v})

# (g x g^3) Delta_2
g_x_g3_Delta2 = {k: [(g[l],) + t for l, r in v for t in g3[r]] for k, v in delta2.items()}
print
print u"( g " + OTIMES + " g^3 ) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in g_x_g3_Delta2.items() if v})

# (g^2 x g^2) Delta_2
g2_x_g2_Delta2 = {k: [s + t for l, r in v for s in g2[l] for t in g2[r]] for k, v in delta2.items()}
print
print u"( g^2 " + OTIMES + " g^2 ) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in g2_x_g2_Delta2.items() if v})

# (g^3 x g) Delta_2
g3_x_g_Delta2 = {k: [t + (g[r],) for l, r in v for t in g3[l]] for k, v in delta2.items()}
print
print u"( g^3 " + OTIMES + " g ) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in g3_x_g_Delta2.items() if v})

# (g x g x g^2) Delta_3
g_x_g_x_g2_Delta3 = {k: [(g[l], g[m]) + t for l, m, r in v for t in g2[r]] for k, v in delta3.items()}
print
print u"( g " + OTIMES + " g " + OTIMES + " g^2 ) " + DELTA + "_3 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in g_x_g_x_g2_Delta3.items() if v})

# (g x g^2 x g) Delta_3
g_x_g2_x_g_Delta3 = {k: [(g[l], ) + t + (g[r], ) for l, m, r in v for t in g2[m]] for k, v in delta3.items()}
print
print u"( g " + OTIMES + " g^2 " + OTIMES + " g ) " + DELTA + "_3 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in g_x_g2_x_g_Delta3.items() if v})

# (g^2 x g x g) Delta_3
g2_x_g_x_g_Delta3 = {k: [t + (g[m], g[r]) for l, m, r in v for t in g2[l]] for k, v in delta3.items()}
print
print u"( g^2 " + OTIMES + " g " + OTIMES + " g ) " + DELTA + "_3 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in g2_x_g_x_g_Delta3.items() if v})

# phi_2
phi_2 = add_maps_mod_2(
    add_maps_mod_2(
        add_maps_mod_2(
            add_maps_mod_2(id_x_id_x_Delta_g3, id_x_Delta_x_id_g3),
            Delta_x_id_x_id_g3),
        add_maps_mod_2(
            add_maps_mod_2(g_x_g3_Delta2, g3_x_g_Delta2),
            g2_x_g2_Delta2)
        ),
    add_maps_mod_2(
        add_maps_mod_2(g_x_g_x_g2_Delta3, g_x_g2_x_g_Delta3),
        g2_x_g_x_g_Delta3)
    )
print
print PHI + u"_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in phi_2.items() if v})


CxCxCxC = tensor(C.groups, C.groups, C.groups, C.groups)
# print CxCxCxC # debug

dCxCxCxC = {}
for k, vs in CxCxCxC.items():
    dCxCxCxC[k] = {}
    for (v1, v2, v3, v4) in vs:
        dv1 = [(v1_i, v2, v3, v4) for v1_i in C.differential[v1]] if v1 in C.differential else []
        dv2 = [(v1, v2_i, v3, v4) for v2_i in C.differential[v2]] if v2 in C.differential else []
        dv3 = [(v1, v2, v3_i, v4) for v3_i in C.differential[v3]] if v3 in C.differential else []
        dv4 = [(v1, v2, v3, v4_i) for v4_i in C.differential[v4]] if v4 in C.differential else []
        if dv1 + dv2 + dv3 + dv4:
            dCxCxCxC[k][(v1, v2, v3, v4)] = dv1 + dv2 + dv3 + dv4

delta4 = {}
g4 = {} # all boundary found in image will become part of g4
for k, v in g.items():
    dim = int(k[1])+3

    img = phi_2[k]
    g4[k] = []
    for chain, bd in dCxCxCxC[dim].items():
        if all([cell in img for cell in bd]):
            g4[k].append(chain)
            for cell in bd:
                img.remove(cell)
    delta4[k] = img

delta4 = {k: [(g_inv[v1], g_inv[v2], g_inv[v3], g_inv[v4]) for (v1, v2, v3, v4) in v] for k, v in delta4.items() if v}

print
print DELTA + u"_4 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta4.items()})

# g^4
print
print u"g^4 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g4.items() if v})
