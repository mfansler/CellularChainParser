# Standard imports
from re import sub
from re import compile
from argparse import ArgumentParser

from itertools import product
from collections import Counter

import numpy

import sys
import codecs
sys.stdout=codecs.getwriter('utf-8')(sys.stdout)

# Local imports
import CellChainParse
from Coalgebra import Coalgebra
from factorize import factorize_recursive as factorize
from support_functions import generate_f_integral, row_reduce_mod2, add_maps_mod_2, derivative, expand_tuple_list, list_mod

__author__ = 'mfansler'
temp_mat = "~transfer-temp.mat"

# chars
DELTA   = u"\u0394"
PARTIAL = u"\u2202"
NABLA   = u"\u2207"
OTIMES  = u"\u2297"
THETA   = u"\u03b8"
PHI     = u"\u03c6"
CHAINPARTIAL = "(1" + OTIMES + PARTIAL + " + " + PARTIAL + OTIMES + "1)"

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


def chain_coproduct(chain, coproduct, simplify=True):

    ps = [p for cell in chain for p, num in coproduct[cell].items() if num % 2]
    return [el for el, num in Counter(ps).items() if num % 2] if simplify else ps


hom_dim_re = compile('h(\d*)_')


def hom_dim(h_element):
    return int(hom_dim_re.match(h_element).group(1))


def differential2_space(AxA, diff_op):

    dAxA = {}
    for dim, vs in AxA.items():
        dAxA[dim] = {}
        for (l, r) in vs:
            dLeft = [(l_i, r) for l_i in diff_op[l]] if l in diff_op else []
            dRight = [(l, r_i) for r_i in diff_op[r]] if r in diff_op else []
            if dLeft + dRight:
                dAxA[dim][(l, r)] = dLeft + dRight

    return dAxA

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

H_gens = {}

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

# generate f: C -> H
f, integrate = generate_f_integral(C, g)

# define Delta g
delta_g = {k: chain_coproduct(v, C.coproduct) for k, v in g.items()}
print
print DELTA + u"g =", format_morphism(delta_g)

print
print DELTA + u"g (unsimplified) =", {k: chain_coproduct(v, C.coproduct, simplify=False) for k, v in g.items()}

factored_delta_g = {k: factorize(v) for k, v in delta_g.items()}
print
print DELTA + u"g (factored) =", factored_delta_g

delta2 = {k: [tuple(map(f, list(t))) for t in tuples] for k, tuples in factored_delta_g.items()}

# flatten delta2 and remove up empty elements
delta2 = {k: [tp_i for tp in tps for tp_i in expand_tuple_list(tp)]for k, tps in delta2.items()}
print
print DELTA + u"_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta2.items()})

# (g x g) Delta2
gxgDelta = {k: [(g_l, g_r) for l, r in v for g_l in g[l] for g_r in g[r]] for k, v in delta2.items()}
print
print u"(g " + OTIMES + " g)" + DELTA + "_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in gxgDelta.items()})

# nabla g^2
nabla_g2 = add_maps_mod_2(gxgDelta, delta_g)
print
print u"(g " + OTIMES + " g)" + DELTA + "_2 + " + DELTA + "g =", format_morphism({k: [format_tuple(t) for t in v] for k, v in nabla_g2.items() if v})

# factored_nabla_g2 = {k: factorize(v) for k, v in nabla_g2.items() if v}
# print
# print u"(g " + OTIMES + " g)" + DELTA + "_2 + " + DELTA + "g (factored) =", factored_nabla_g2

# g^2
g2 = {k: integrate(vs) for k, vs in nabla_g2.items()}
g2 = {k: [tp_i for tp in tps for tp_i in expand_tuple_list(tp)]for k, tps in g2.items() if tps}
print
print u"g^2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g2.items() if v})

nabla_g2_computed = {k: [exp_tp for (l, r) in derivative(v, C) for exp_tp in expand_tuple_list((l, r)) if l and r] for k, v in g2.items() if v}
nabla_g2_computed = {k: list_mod(vs, modulus=2) for k, vs in nabla_g2_computed.items()}
print
print NABLA + u" g^2 =", format_morphism(nabla_g2_computed)

# VERIFY: DELTA g = (g x g) DELTA_2 + NABLA g^2

print
print u"(g " + OTIMES + " g)" + DELTA + "_2 + " + DELTA + "g + " + NABLA + "g^2 = 0 ? ",
print "FALSE!" if any(add_maps_mod_2(nabla_g2_computed, nabla_g2).values()) else "TRUE!"

# -------------------------------------------- #
exit()

# CxC
CxC = tensor(C.groups, C.groups)

# dCxC
dCxC = differential2_space(CxC, C.differential)

# d(Delta g)
d_Delta_g = {k: [] for k in g.keys()}
for k, vs in Delta_g.items():

    for (l, r) in vs:
        dLeft = [(l_i, r) for l_i in C.differential[l]] if l in C.differential else []
        dRight = [(l, r_i) for r_i in C.differential[r]] if r in C.differential else []
        if dLeft + dRight:
            d_Delta_g[k] += dLeft + dRight

d_Delta_g = {k: list_mod(v, 2) for k, v in d_Delta_g.items()}
print u"d" + DELTA + u"g =", format_morphism(d_Delta_g)
print Delta_g

# for k, vs in CxC.items():
#     dCxC[k] = {}
#     for (l, r) in vs:
#         dLeft = [(l_i, r) for l_i in C.differential[l]] if l in C.differential else []
#         dRight = [(l, r_i) for r_i in C.differential[r]] if r in C.differential else []
#         if dLeft + dRight:
#             dCxC[k][(l, r)] = dLeft + dRight

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
def H_to_CxC_0():
    return ({h: cxc} for dim, hs in H.items() for h in hs for cxc in CxC[dim])
#print "size[H->CxC] = ", len(H_to_CxC_0)

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

# convert into basis
def H_to_dCxC_1_vecs():
    return ((hs, get_vector_in_basis(h_to_cxcs, H_to_CxC_0())) for (hs, h_to_cxcs) in H_to_dCxC_1())

print "DEBUG: generating matrices"
X_img = numpy.array([vec for (_, vec) in g_x_g_H_to_HxH_0_vecs()], dtype=numpy.int8).transpose()
X_ker = numpy.array([vec for (_, vec) in H_to_dCxC_1_vecs()], dtype=numpy.int8).transpose()
y = numpy.array([delta_g_vec], dtype=numpy.int8).transpose()

img_size = X_img.shape[1]
input_matrix = numpy.append(numpy.append(X_img, X_ker, axis=1), y, axis=1)
print "DEBUG: matrices generated"

print "DEBUG: row reducing matrix"
sols_mat = row_reduce_mod2(input_matrix, -1)
print "DEBUG: matrix row reduced"

print "DEBUG: extracting results"
delta2 = {k: [] for k in g.keys()}
g2 = {k: [] for k in g.keys()}

for i in [i for i in numpy.nonzero(sols_mat[:, -1])[0]]:
    # get leftmost non-zero column
    j = numpy.nonzero(sols_mat[i, :])[0][0]
    if j < img_size:
        for h, hxh in list(g_x_g_H_to_HxH_0_vecs())[j][0].items():
            delta2[h] = delta2[h] + [hxh] if h in delta2 else [hxh]
        #print g_x_g_H_to_HxH_0_vecs[j][0]
    else:
        for h, cxc in list(H_to_dCxC_1())[j - img_size][0].items():
            g2[h] = g2[h] + [cxc] if h in g2 else [cxc]
        #print H_to_dCxC_1[j - len(g_x_g_H_to_HxH_0_vecs)]

# Clean up
del sols_mat, input_matrix, X_img, X_ker, y, delta_g_vec, HxH, CxC, dCxC

print
print DELTA + u"_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta2.items()})
print DELTA
exit()
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

# Clean up
del sum_Delta_g, nabla_g2, gxgDelta

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

print "DEBUG: generating CxCxC"
CxCxC = tensor(C.groups, C.groups, C.groups)
print "DEBUG: CxCxC size ", {k: len(vs) for k, vs in CxCxC.items()}


print "DEBUG: generating dCxCxC"
dCxCxC = {}
for k, vs in CxCxC.items():
    dCxCxC[k] = {}
    for (l, m, r) in vs:
        dLeft = [(l_i, m, r) for l_i in C.differential[l]] if l in C.differential else []
        dMiddle = [(l, m_i, r) for m_i in C.differential[m]] if m in C.differential else []
        dRight = [(l, m, r_i) for r_i in C.differential[r]] if r in C.differential else []
        if dLeft + dMiddle + dRight:
            dCxCxC[k][(l, m, r)] = dLeft + dMiddle + dRight

delta3 = {k: [] for k in g.keys()}
g3 = {k: [] for k in g.keys()}

# for k, v in g.items():
#     dim = int(k[1])+2
#
#     img = phi_1[k]
#     g3[k] = []
#     for chain, bd in dCxCxC[dim].items():
#         if all([cell in img for cell in bd]):
#             g3[k].append(chain)
#             for cell in bd:
#                 img.remove(cell)
#     delta3[k] = img
#
# delta3 = {k: [(g_inv[l], g_inv[m], g_inv[r]) for (l, m, r) in v] for k, v in delta3.items() if v}

# basis for vector space
def H_to_CxCxC_1():
    return ({h: cxcxc} for dim, hs in H.items() for h in hs for cxcxc in CxCxC[dim + 1])

print "DEBUG: generating phi_1 vector"
phi_1_vec = get_vector_in_basis(phi_1, H_to_CxCxC_1())

print "DEBUG: generating HxHxH"
HxHxH = tensor(H, H, H)
print "DEBUG: HxHxH size ", {k: len(vs) for k, vs in HxHxH.items()}

def H_to_HxHxH_1():
    print "\tDEBUG: enter H_to_HxHxH_1 generator"
    return ({h: hxhxh} for dim, hs in H.items() for h in hs for hxhxh in HxHxH[dim + 1])

def gxgxg_H_to_HxHxH_1():
    print "\tDEBUG: enter gxgxg_H_to_HxHxH_1 generator"
    return ((hs, {h: [(l, m, r) for l in g[h_l] for m in g[h_m] for r in g[h_r]]
                  for h, (h_l, h_m, h_r) in hs.items()}) for hs in H_to_HxHxH_1())

def gxgxg_H_to_HxHxH_1_vecs():
    print "\tDEBUG: enter gxgxg_H_to_HxHxH_1_vecs generator"
    return ((hs, get_vector_in_basis(h_to_cxcxcs, H_to_CxCxC_1())) for (hs, h_to_cxcxcs) in gxgxg_H_to_HxHxH_1())


# generate all components in the Hom_0(H, dCxCxC) space
def H_to_dCxCxC_1():
    print "\tDEBUG: enter H_to_dCxCxC_1 generator"
    return (({h: cxcxc}, {h: dcxcxc}) for dim, hs in H.items() for h in hs for cxcxc, dcxcxc in dCxCxC[dim+2].items() if dcxcxc)

def H_to_dCxCxC_1_vecs():
    print "\tDEBUG: enter H_to_dCxCxC_1_vecs generator"
    return ((hs, get_vector_in_basis(h_to_cxcxcs, H_to_CxCxC_1())) for (hs, h_to_cxcxcs) in H_to_dCxCxC_1())

print "DEBUG: Generating Image (h --> CxCxC) matrix"
X_img = numpy.array([vec for (_, vec) in gxgxg_H_to_HxHxH_1_vecs()], dtype=numpy.int8).transpose()
img_size = X_img.shape[1]

print "DEBUG: Generating Kernel (h --> CxCxC) matrix"
#X_ker = numpy.array([vec for (_, vec) in H_to_dCxCxC_1_vecs()], dtype=numpy.int8).transpose()

print "DEBUG: Generating Phi (h --> CxCxC) vector"
y = numpy.array([phi_1_vec], dtype=numpy.int8).transpose()

print "DEBUG: appending matrices"
#input_matrix = numpy.append(numpy.append(X_img, X_ker, axis=1), y, axis=1)
input_matrix = numpy.append(X_img, y, axis=1)

print "DEBUG: Row reducing matrix"
sols_mat = row_reduce_mod2(input_matrix, -1)

print "DEBUG: Extracting solution"
gxgxg_H_to_HxHxH_1_vecs_list = list(gxgxg_H_to_HxHxH_1_vecs())
for i in [i for i in numpy.nonzero(sols_mat[:, -1])[0]]:
    # get leftmost non-zero column
    j = numpy.nonzero(sols_mat[i, :])[0][0]
    #print j
    if j < img_size:
        for h, hxhxh in gxgxg_H_to_HxHxH_1_vecs_list[j][0].items():
            delta3[h] = delta3[h] + [hxhxh] if h in delta3 else [hxhxh]
        #print g_x_g_H_to_HxH_0_vecs[i]
    # else:
    #     for h, cxcxc in list(H_to_dCxCxC_1())[j - img_size][0].items():
    #         g3[h] = g3[h] + [cxcxc] if h in g3 else [cxcxc]
        #print H_to_dCxC_1[i - len(g_x_g_H_to_HxH_0_vecs)]


print
print DELTA + u"_3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta3.items()})

# g^3
print
print u"g^3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g3.items() if v})

# Clean up
del X_img, y, phi_1_vec, z_1, HxHxH, CxCxC, dCxCxC, input_matrix, sols_mat#, X_ker

# VERIFY: phi_1 = (g x g x g) DELTA_3 + NABLA g^3

# (g x g x g) Delta_3
gxgxgDelta3 = {k: [(g_l, g_m, g_r) for l, m, r in v for g_l in g[l] for g_m in g[m] for g_r in g[r]] for k, v in delta3.items()}
print
print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in gxgxgDelta3.items()})

# NABLA g^3
nabla_g3 = {}
for k, vs in g3.items():
    nabla_g3[k] = []
    for (l, m, r) in vs:
        dLeft   = [(l_i, m, r) for l_i in C.differential[l]] if l in C.differential else []
        dMiddle = [(l, m_i, r) for m_i in C.differential[m]] if m in C.differential else []
        dRight  = [(l, m, r_i) for r_i in C.differential[r]] if r in C.differential else []
        if dLeft + dMiddle + dRight:
            nabla_g3[k][(l, m, r)] = dLeft + dMiddle + dRight

print
print NABLA + u" g^3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in nabla_g3.items() if v})

# # (g x g x g) DELTA_3 + NABLA g^3
# sum_phi_1 = add_maps_mod_2(gxgxgDelta3, nabla_g3)
# print
# print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 + " + NABLA + " g^3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in sum_phi_1.items()})

# # phi_1 = (g x g x g) DELTA_2 + NABLA g^2
# print
# print PHI + u"_1 = (g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 + " + NABLA + " g^3 : ",
# print all([vs == [] for vs in add_maps_mod_2(sum_phi_1, phi_1).values()])

# (g x g x g) DELTA_3 + NABLA g^3
sum_nabla_g3 = add_maps_mod_2(gxgxgDelta3, phi_1)
print
print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 + " + PHI + "_1 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in sum_nabla_g3.items()})


#--------------------------------------------#

exit()
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
phi_2 = reduce(add_maps_mod_2, [id_x_id_x_Delta_g3, id_x_Delta_x_id_g3, Delta_x_id_x_id_g3, g_x_g3_Delta2,
                                g3_x_g_Delta2, g2_x_g2_Delta2, g_x_g_x_g2_Delta3, g_x_g2_x_g_Delta3, g2_x_g_x_g_Delta3], {})
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
