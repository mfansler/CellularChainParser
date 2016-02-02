# Standard imports
from subprocess import check_output, CalledProcessError
from os import remove as rm

from re import sub
from re import compile
from argparse import ArgumentParser

from itertools import product

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
        return str(t)


def format_sum(obj):
    if obj is None:
        return "0"
    elif type(obj) is dict:
        single = [format_tuple(k) for k, v in obj.items() if v == 1]
        multiple = [u"{}*({})".format(v, format_tuple(k)) for k, v in obj.items() if v > 1]
        return " + ".join(single + multiple)
    elif type(obj) is list:
        return "(" + u" + ".join([o for o in obj]) + ")"
    else:
        return obj


def format_morphism(m):
    return u"\n\t+ ".join([u"{}{}_{{{}}}".format('(' + format_morphism(v) + ')' if type(v) is dict else format_sum(v), PARTIAL, k) for k, v in m.items()])


def compare_incidences(x, y):
    return x[1] - y[1] if x[1] != y[1] else x[0] - y[0]


def tensor(a, b):

    a_max = max(a.keys())
    b_max = max(b.keys())
    tensor_groups = {i: [] for i in range(a_max*b_max+1)}

    for m in range(a_max + 1):
        for n in range(b_max + 1):
            tensor_groups[m+n] += product(a[m], b[n])

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


H = {}
offset = 9 + len(dims)
for n, k in enumerate(dims):
    #print "debug: ", C.groups[n]
    H[n] = [[C.groups[n][int(j)] for j in compile('\[(\d+)\]').findall(lines[offset + i])] for i in range(k)]
    #print "debug: ", lines[offset], k
    offset += k + 1

print "H = H*(C) = {",
print ", ".join(["h{}_{} = {}".format(n, i, gen) for n, gens in H.items() for i, gen in enumerate(gens)]),
print "}"

# Define g
g = {"h{}_{}".format(n, i): gen[0] for n, gens in H.items() for i, gen in enumerate(gens) if gen}
print
print "g = ", format_morphism(g)

# Define g inverse
g_inv = {v: k for k, v in g.items()}

alpha = {}
beta = {}

# K2 = Delta
alpha[THETA+'2'] = C.coproduct

# J1 -> g

beta['f1'] = g

# J2: theta2 f1 -> Delta g
beta[THETA+'2f1'] = {}
for k, v in g.items():
    beta[THETA+'2f1'][k] = '(' + format_sum(C.coproduct[v]) + ')'

# define Delta g
print
print DELTA + u"g =", format_morphism(beta[THETA+'2f1'])
#print "beta = ", format_morphism(beta)

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
for k, v in g.items():
    dim = int(k[1])+1
    img = C.coproduct[v].keys()
    g2[k] = []
    for chain, bd in dCxC[dim].items():
        if all([cell in img for cell in bd]):
            g2[k].append(chain)
            for cell in bd:
                img.remove(cell)
    delta2[k] = img

delta2 = {k: [(g_inv[l], g_inv[r]) for (l, r) in v] for k, v in delta2.items()}
print
print DELTA + u"_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in delta2.items()})


# (g x g) Delta
gxgDelta = {k: [(g[l], g[r]) for l, r in v] for k, v in delta2.items()}
print
print u"(g " + OTIMES + " g)" + DELTA + "_2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in gxgDelta.items()})

# f^2
print
print u"g^2 =", format_morphism({k: [format_tuple(t) for t in v] for k, v in g2.items() if v})

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


# CxCxC = tensor(C.groups, C.groups, C.groups)
#
# dCxCxC = {}
# for k, vs in CxCxC.items():
#     dCxCxC[k] = {}
#     for (l, r) in vs:
#         dLeft = [(l_i, r) for l_i in C.differential[l]] if l in C.differential else []
#         dRight = [(l, r_i) for r_i in C.differential[r]] if r in C.differential else []
#         if dLeft + dRight:
#             dCxCxC[k][(l, r)] = dLeft + dRight

# print CxC # debug
