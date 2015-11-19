# Standard imports
from subprocess import check_output, CalledProcessError
from os import remove as rm

from re import sub
from re import compile
from argparse import ArgumentParser


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
        return '(', " + ".join(v), ')'
    else:
        return obj

def format_morphism(m):
    return u" + ".join([u"{}{}_{{{}}}".format('(' + format_morphism(v) + ')' if type(v) is dict else v, PARTIAL, k) for k, v in m.items()])

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
    for entry in ["{} {} {}".format(l, r, v % 2) for (l, r), v in entries.iteritems()]:
        print >> scratch, entry

scratch.close()

try:
    chomp_results = check_output(["chomp-matrix", temp_mat, "-gp2"])
    print chomp_results
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
    H[n] = [[C.groups[n][int(j)] for j in compile('\[(\d+)\]').findall(lines[offset + i])] for i in range(k)]
    #print "debug: ", lines[offset], k
    offset += k + 1

print "H = H*(C) = {",
print ", ".join(["h{}_{} = {}".format(n, i, gen) for n, gens in H.items() for i, gen in enumerate(gens)]),
print "}"


# Define g
g = {"h{}_{}".format(n, i): gen[0] for n, gens in H.items() for i, gen in enumerate(gens)}

print "g = ", format_morphism(g)

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

print "beta = ", format_morphism(beta)

# identify theta2 f1 -> Delta g

