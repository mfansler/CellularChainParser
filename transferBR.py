# Standard imports
from re import sub
from re import compile
from argparse import ArgumentParser

from itertools import product
from collections import Counter

import numpy
import scipy.sparse as sp

import sys
import codecs
sys.stdout=codecs.getwriter('utf-8')(sys.stdout)

# Local imports
import CellChainParse
from Coalgebra import Coalgebra
from support_functions import *

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
        return u" \u2297 ".join([format_sum(x) if type(x) is list else x for x in t])
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

differential = {n: C.incidence_matrix(n, sparse=False) for n in range(1, C.topDimension() + 1)}
delta_c = {k: [c for c, i in v.items() if i % 2] for k, v in C.coproduct.items()}

"""
Checking Coassociativity on Delta_C
"""

# (1 x Delta) Delta
id_x_Delta_Delta = {k: [(l,) + r_cp for (l, r) in v for r_cp in delta_c[r]] for k, v in delta_c.items()}
# print
# print u"(1 " + OTIMES + " " + DELTA + ") " + DELTA + " =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in id_x_Delta_Delta.items() if v})

# (Delta x 1) Delta
Delta_x_id_Delta = {k: [l_cp + (r,) for (l, r) in v for l_cp in delta_c[l]] for k, v in delta_c.items()}
# print
# print u"(" + DELTA + " " + OTIMES + " 1) " + DELTA + " =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in Delta_x_id_Delta.items() if v})

# DeltaC = (1 x Delta_C + Delta_C x 1) Delta_C
id_x_Delta_Delta_x_id_Delta = add_maps_mod_2(id_x_Delta_Delta, Delta_x_id_Delta)
id_x_Delta_Delta_x_id_Delta = {k: list_mod(ls, modulus=2) for k, ls in expand_map_all(id_x_Delta_Delta_x_id_Delta).items()}
print DELTA + "_c is co-associative?", not any(id_x_Delta_Delta_x_id_Delta.values())

if any(id_x_Delta_Delta_x_id_Delta.values()):
    print u"\n(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + " =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in id_x_Delta_Delta_x_id_Delta.items() if v})
    #print "(raw) =", id_x_Delta_Delta_x_id_Delta
    # factored_id_x_Delta_Delta_id_Delta = {k: factorize(v) for k, v in id_x_Delta_Delta_id_Delta.items()}
    # print factored_id_x_Delta_Delta_id_Delta


"""
COMPUTE HOMOLOGY
"""

H_gens = {}

# Manually entering results from SageMath for basis for homology
H_gens[0] = [['v_{1}']]
H_gens[1] = [['m_{11}', 'm_{4}'], ['c_{3}', 'c_{7}'], ['m_{6}', 'm_{9}']]
H_gens[2] = [['t_{1}', 't_{2}', 't_{3}', 't_{4}'], ['t_{5}', 't_{6}', 't_{7}', 't_{8}']]
#H_gens[3] = []

H = {dim: ["h{}_{}".format(dim, i) for i, gen in enumerate(gens)] for dim, gens in H_gens.items()}
print "\nH = H*(C) = ", H


"""
DEFINE MAPS BETWEEN CHAINS AND HOMOLOGY
"""

# Define g
g = {"h{}_{}".format(dim, i): gen for dim, gens in H_gens.items() for i, gen in enumerate(gens) if gen}
print
print "g = ", format_morphism(g)

# generate f: C -> H
f, integrate = generate_f_integral(C, g)

"""
COMPUTE Delta_2, g^2
"""

# define Delta g
delta_g = {k: chain_coproduct(v, C.coproduct) for k, v in g.items()}
print
print DELTA + u"g =", format_morphism(delta_g)

print
print DELTA + u"g (unsimplified) =", {k: chain_coproduct(v, C.coproduct, simplify=False) for k, v in g.items()}

factored_delta_g = {k: factorize_cycles(v, C) for k, v in delta_g.items()}
print
print DELTA + u"g (factored) =", format_morphism(factored_delta_g)


delta2 = {k: [tuple(map(f, list(t))) for t in tuples] for k, tuples in factored_delta_g.items()}

# flatten delta2 and remove up empty elements
#delta2 = {k: [tp_i for tp in tps for tp_i in expand_tuple_list(tp)]for k, tps in delta2.items()}
delta2 = chain_map_mod(expand_map_all(delta2))
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

# g^2
g2 = {k: integrate(vs) for k, vs in nabla_g2.items()}
g2 = {k: [tp_i for tp in tps for tp_i in expand_tuple_list(tp)]for k, tps in g2.items()}
print
print u"g^2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g2.items() if v})

"""
VERIFY CONSISTENCY OF DELTA2, g^2 RESULTS
"""

nabla_g2_computed = {k: [exp_tp for (l, r) in derivative(v, C) for exp_tp in expand_tuple_list((l, r)) if l and r] for k, v in g2.items() if v}
nabla_g2_computed = {k: list_mod(vs, modulus=2) for k, vs in nabla_g2_computed.items()}
print
print NABLA + u" g^2 =", format_morphism(nabla_g2_computed)

print u"\n(g " + OTIMES + " g)" + DELTA + "_2 + " + DELTA + "g + " + NABLA + "g^2 = 0 ? ",
print not any(add_maps_mod_2(nabla_g2_computed, nabla_g2).values())

# -------------------------------------------- #

"""
VERIFY DELTA2 COASSOCIATIVITY
"""

# (1 x Delta_2) Delta_2
id_x_Delta2_Delta2 = {k: [(l,) + r_cp for (l, r) in v for r_cp in delta2[r]] for k, v in delta2.items()}
# print
# print u"(1 " + OTIMES + " " + DELTA + "_2) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in id_x_Delta2_Delta2.items() if v})

# (Delta_2 x 1) Delta_2
Delta2_x_id_Delta2 = {k: [l_cp + (r,) for (l, r) in v for l_cp in delta2[l]] for k, v in delta2.items()}
# print
# print u"(" + DELTA + "_2 " + OTIMES + " 1) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in Delta2_x_id_Delta2.items() if v})

# z_1 = (1 x Delta_2 + Delta_2 x 1) Delta_2
z_1 = add_maps_mod_2(id_x_Delta2_Delta2, Delta2_x_id_Delta2)
# print
# print u"z_1 = (1 " + OTIMES + " " + DELTA + "_2 + " + DELTA + "_2 " + OTIMES + " 1) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in z_1.items() if v})
print
print DELTA + "_2 is co-associative?", not any(z_1.values())

"""
COMPUTE DELTA_C_3
"""

# #(1 x Delta + Delta x 1) Delta g
# id_x_Delta_Delta_id_Delta_g = {k: list_mod([tp for c in v for tp in id_x_Delta_Delta_x_id_Delta[c]], 2) for k, v in g.items()}
# print
# print u"(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + "g =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in id_x_Delta_Delta_id_Delta_g.items() if v})

# Delta_c3
delta_c3 = integrate(id_x_Delta_Delta_x_id_Delta)
delta_c3 = chain_map_mod(expand_map_all(delta_c3))
print
print DELTA + u"_C3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta_c3.items() if v})

# verify consistency
nabla_delta_c3_computed = derivative(delta_c3, C)
nabla_delta_c3_computed = chain_map_mod(expand_map_all(nabla_delta_c3_computed))

print
print NABLA + DELTA + u"_C3 =", format_morphism(nabla_delta_c3_computed)

print u"\n(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + " + " + NABLA + DELTA + u"_C3 = 0 ? ",
print not any(add_maps_mod_2(id_x_Delta_Delta_x_id_Delta, nabla_delta_c3_computed).values())

if any(add_maps_mod_2(id_x_Delta_Delta_x_id_Delta, nabla_delta_c3_computed).values()):
    print u"\n(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + " + " + NABLA + DELTA + u"_C3 =",
    print format_morphism({k: [format_tuple(t) for t in v] for k, v in add_maps_mod_2(id_x_Delta_Delta_x_id_Delta, nabla_delta_c3_computed).items() if v})

"""
COMPUTE DELTA3, g^3
"""

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

# delta_c3_g
delta_c3_g = {k: [tp for v in vs if v in delta_c3 for tp in delta_c3[v]] for k, vs in g.items()}
print
print DELTA + "_c3 g =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in delta_c3_g.items() if v})

# phi_1 = (1 x Delta + Delta x 1) g^2 + (g x g^2 + g^2 x g) Delta_2 + Delta_C3 g
phi_1 = reduce(add_maps_mod_2, [g_x_g2_Delta2, g2_x_g_Delta2, id_x_Delta_g2, Delta_x_id_g2, delta_c3_g], {})
print
print PHI + u"_1 = (g " + OTIMES + " g^2 + g^2 " + OTIMES + " g) " + DELTA + "_2 +",
print "(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) g^2 + " + DELTA + "_c3 =",
print format_morphism({k: [format_tuple(t) for t in v] for k, v in phi_1.items() if v})
#print PHI + u"_1 (raw) =", phi_1

# Nabla phi_1 == 0 ? (Verify consistency)
nabla_phi_1 = {k: [(l, m, r) for (l, m, r) in derivative(v, C) if l and m and r] for k, v in phi_1.items() if v}
nabla_phi_1 = chain_map_mod(expand_map_all(nabla_phi_1))
print
print NABLA + PHI + u"_1 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in nabla_phi_1.items()})

# factor phi_1
factored_phi_1 = {k: factorize_cycles(v, C) for k, v in phi_1.items() if v}

print
print PHI + u"_1 (factored) =", factored_phi_1

delta3 = {k: [tuple(map(f, list(t))) for t in tuples] for k, tuples in factored_phi_1.items()}

# flatten delta3 and remove up empty elements
delta3 = chain_map_mod(expand_map_all(delta3))
#delta3 = {k: [tp_i for tp in tps for tp_i in expand_tuple_list(tp)]for k, tps in delta3.items()}
print
print DELTA + u"_3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta3.items()})

# (g x g x g) Delta3
gxgxg_delta3 = {k: [(g_l, g_m, g_r) for l, m, r in v for g_l in g[l] for g_m in g[m] for g_r in g[r]] for k, v in delta3.items()}
print
print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in gxgxg_delta3.items()})

# Nabla phi_1 == 0 ? (Verify consistency)
nabla_gxgxg_delta3 = {k: [(l, m, r) for (l, m, r) in derivative(v, C) if l and m and r] for k, v in gxgxg_delta3.items() if v}
nabla_gxgxg_delta3 = chain_map_mod(expand_map_all(nabla_gxgxg_delta3))
print
print NABLA + u"(g x g x g) " + DELTA + u"^3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in nabla_gxgxg_delta3.items()})


# nabla g^3
nabla_g3 = add_maps_mod_2(gxgxg_delta3, phi_1)
print
print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 + " + PHI + "_1 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in nabla_g3.items() if v})

# g^3
g3 = {k: integrate(vs) for k, vs in nabla_g3.items()}
#g3 = {k: [tp_i for tp in tps for tp_i in expand_tuple_list(tp)]for k, tps in g3.items() if tps}
g3 = chain_map_mod(expand_map_all(g3))
print
print u"g^3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g3.items() if v})

"""
VERIFY CONSISTENCY OF phi_1, Delta_3, and g^3
"""

nabla_g3_computed = {k: [(l, m, r) for (l, m, r) in derivative(v, C) if l and m and r] for k, v in g3.items() if v}
nabla_g3_computed = chain_map_mod(expand_map_all(nabla_g3_computed))
print
print NABLA + u" g^3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in nabla_g3_computed.items()})

print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 + " + PHI + "_1 + " + NABLA + "g^3 = 0 ? ",
print not any(reduce(add_maps_mod_2, [gxgxg_delta3, nabla_g3_computed, phi_1], {}).values())

if any(reduce(add_maps_mod_2, [gxgxg_delta3, nabla_g3_computed, phi_1], {}).values()):
    print "\t", reduce(add_maps_mod_2, [gxgxg_delta3, nabla_g3_computed, phi_1], {})


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
