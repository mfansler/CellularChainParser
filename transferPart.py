# Standard imports
from subprocess import check_output, CalledProcessError
from os import remove as rm

from re import sub, compile
from argparse import ArgumentParser, FileType

from itertools import product
from collections import Counter

import numpy

import sys
import codecs
sys.stdout=codecs.getwriter('utf-8')(sys.stdout)

# Local imports
import CellChainParse
from Coalgebra import Coalgebra
from factorize import factorize
from support_functions import *
from formatting import *

__author__ = 'mfansler'
temp_mat = "~transfer-temp.mat"


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
argparser.add_argument('--hgroups', '-hg', dest='homology_groups', type=FileType('r'), help="File containing homology groups")
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

# convert coproduct to map form
delta_c = {}
for from_cell, to_cells in C.coproduct.iteritems():
    delta_c[from_cell] = [cell for cell, count in to_cells.iteritems() if count % 2]

"""
Check Coassociativity on Delta_C
"""

# (1 x Delta) Delta
id_x_Delta_Delta = {}
for c, cxc in delta_c.iteritems():
    id_x_Delta_Delta[c] = [(l,) + r_cp for (l, r) in cxc for r_cp in delta_c[r]]

# (Delta x 1) Delta
Delta_x_id_Delta = {k: [l_cp + (r,) for (l, r) in v for l_cp in delta_c[l]] for k, v in delta_c.items()}

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

if not args.homology_groups:

    # create temporary file for CHomP
    scratch = open(temp_mat, 'w+')

    differential = {n: C.incidence_matrix(n, sparse=False) for n in range(1, C.topDimension() + 1)}

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

else:

    try:
        H_gens = eval(args.homology_groups.read())
    except Exception as e:
        print "Error: Unable to load homology groups!"
        argparser.print_help()
        raise SystemExit

    args.homology_groups.close()

    # verify that the provided homology generators are valid for this cell complex
    for i, group in H_gens.iteritems():
        for cells in group:
            if not all([cell in C.groups[i] for cell in cells]):
                print "Error: Invalid homology groups provided! Cells in generators not found in the complex."
                raise SystemExit

            d_cells = list_mod(derivative(cells, C))
            if d_cells:
                print "Error: Group generator in homology is not a cycle in the cell complex."
                print "Dimension =", i, "; Generator =", cells
                print "Derivative =", d_cells
                raise SystemExit

H = {dim: ["h{}_{}".format(dim, i) for i, gen in enumerate(gens)] for dim, gens in H_gens.iteritems()}
print "\nH = H*(C) = ", H


"""
DEFINE MAPS BETWEEN CHAINS AND HOMOLOGY
"""

# Define g
g = {}
for dim, gens in H_gens.iteritems():
    for i, gen in enumerate(gens):
        if gen:
            g["h{}_{}".format(dim, i)] = gen

# g = {"h{}_{}".format(dim, i): gen for dim, gens in H_gens.iteritems() for i, gen in enumerate(gens) if gen}
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

factored_delta_g = {k: factorize(v, C) for k, v in delta_g.items()}
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

"""
VERIFY CONSISTENCY OF DELTA2, g^2 RESULTS
"""

nabla_g2_computed = {k: [exp_tp for (l, r) in derivative(v, C) for exp_tp in expand_tuple_list((l, r)) if l and r] for k, v in g2.items() if v}
nabla_g2_computed = {k: list_mod(vs, modulus=2) for k, vs in nabla_g2_computed.items()}
print
print NABLA + u" g^2 =", format_morphism(nabla_g2_computed)

print u"\n(g " + OTIMES + " g)" + DELTA + "_2 + " + DELTA + "g + " + NABLA + "g^2 = 0 ? ",
print not any(add_maps_mod_2(nabla_g2_computed, nabla_g2).values())


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

# phi_1 = (1 x Delta + Delta x 1) g^2 + (g x g^2 + g^2 x g) Delta_2
phi_1 = reduce(add_maps_mod_2, [g_x_g2_Delta2, g2_x_g_Delta2, id_x_Delta_g2, Delta_x_id_g2], {})
print
print PHI + u"_1 = (g " + OTIMES + " g^2 + g^2 " + OTIMES + " g) " + DELTA + "_2 =",  format_morphism({k: [format_tuple(t) for t in v] for k, v in phi_1.items() if v})

# factor phi_1
factored_phi_1 = {k: factorize_cycles(v, C) for k, v in phi_1.items() if v}
print
print PHI + u"_1 (factored) =", factored_phi_1

delta3 = {k: [tuple(map(f, list(t))) for t in tuples] for k, tuples in factored_phi_1.items()}

# flatten delta3 and remove up empty elements
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
