# Standard imports
from subprocess import check_output, CalledProcessError
from os import remove as rm

from re import sub, compile
from argparse import ArgumentParser

from itertools import product
from collections import Counter

import numpy

# Local imports
import CellChainParse
from Coalgebra import Coalgebra
from factorize import factorize_recursive as factorize
from support_functions import generate_f_integral, row_reduce_mod2, add_maps_mod_2, derivative, expand_tuple_list

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
        return u" \u2297 ".join([format_sum(el) for el in t])
    else:
        return unicode(t)


def format_sum(obj):
    if obj is None:
        return "0"
    if type(obj) is dict:
        single = [format_tuple(k) for k, v in obj.items() if v == 1]
        multiple = [u"{}*({})".format(v, format_tuple(k)) for k, v in obj.items() if v > 1]
        return u" + ".join(single + multiple)
    if type(obj) is list:
        return u" + ".join([format_tuple(o) for o in obj])

    return obj


def format_morphism(m):
    return u"\n\t+ ".join([u"{}{}_{{{}}}".format(format_morphism(v) if type(v) is dict else '(' + format_sum(v) + ')', PARTIAL, k) for k, v in m.items()])


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


def get_vector_in_basis(el, basis):

    # el = [{}, {}]
    # basis = [ {'h0_0': ['a','b','c',..,'z']}, {'h0_1': ['b', 'c', ...]}, ..., {}]
    return [1 if k in el and v in el[k] else 0 for b in basis for k, v in b.items()]


hom_dim_re = compile('h(\d*)_')


def hom_dim(h_element):
    return int(hom_dim_re.match(h_element).group(1))


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
    #print chomp_results
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

# g^2
g2 = {k: integrate(vs) for k, vs in nabla_g2.items()}
print
print u"g^2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g2.items() if v})

print
print NABLA + u" g^2 =", format_morphism({k: [(l, r) for (l, r) in derivative(v, C) if l and r] for k, v in g2.items() if v})

# VERIFY: DELTA g = (g x g) DELTA_2 + NABLA g^2

# -------------------------------------------- #


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

# factor phi_1
factored_phi_1 = {k: factorize(v) for k, v in phi_1.items() if v}
print
print PHI + u"_1 (factored) =", factored_phi_1

delta3 = {k: [tuple(map(f, list(t))) for t in tuples] for k, tuples in factored_phi_1.items()}

# flatten delta2 and remove up empty elements
delta3 = {k: [tp_i for tp in tps for tp_i in expand_tuple_list(tp)]for k, tps in delta3.items()}
print
print DELTA + u"_3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta3.items()})

# (g x g x g) Delta3
gxgxg_delta3 = {k: [(g_l, g_m, g_r) for l, m, r in v for g_l in g[l] for g_m in g[m] for g_r in g[r]] for k, v in delta3.items()}
print
print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in gxgxg_delta3.items()})

# nabla g^3
nabla_g3 = add_maps_mod_2(gxgxg_delta3, phi_1)
print
print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 + " + PHI + "_1 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in nabla_g3.items() if v})

# g^3
g3 = {k: integrate(vs) for k, vs in nabla_g3.items()}
print
print u"g^3 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g3.items() if v})

print
print NABLA + u" g^3 =", format_morphism({k: [(l, m, r) for (l, m, r) in derivative(v, C) if l and m and r] for k, v in g3.items() if v})
