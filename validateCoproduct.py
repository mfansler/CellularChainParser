__author__ = 'mfansler'

import sys
from collections import Counter

import CellChainParse

DELTA   = u"\u0394"
PARTIAL = u"\u2202"
OTIMES  = u"\u2297"
CHAINPARTIAL = "(1" + OTIMES + PARTIAL + " + " + PARTIAL + OTIMES + "1)"

def reduceModN(obj, n = 2):
    return {k: 1 for k, v in obj.items() if v % n == 1}

def formatTuple(t):
    if type(t) is tuple:
        return u" \u2297 ".join(list(t))
    else:
        return str(t)

def formatSum(dict_obj):
    if dict_obj is None:
        return "0"
    single = [formatTuple(k) for k, v in dict_obj.items() if v == 1]
    multiple = [u"{}*({})".format(v, formatTuple(k)) for k, v in dict_obj.items() if v > 1]
    return " + ".join(single + multiple)

# Check if input file is specified
if len(sys.argv) == 2:
    f = open(sys.argv[1])
    data = f.read()
    f.close()
    result = CellChainParse.parse(data)

    if not result:
        raise SystemExit

    topDimension = max([int(g) for g in result["groups"]])

    differential = {}

    for n in range(1, topDimension + 1):

        differential[n] = {}

        for j, face in enumerate(result["groups"][n]):

            if face not in result["differentials"]:
                print "Warning: Boundary of {} not specified; null is assumed".format(face)
                continue

            for edge, count in result["differentials"][face].iteritems():

                i = result["groups"][n-1].index(edge)
                differential[n][(i, j)] = count

    allValid = True;
    for n in range(1, topDimension+1):
        for i in result["groups"][n]:

            # Compute and print \Delta \partial (X)
            boundary = result["differentials"][i] if i in result["differentials"] else None
            lhs = DELTA + PARTIAL + str(i)
            print u"{} = {} ({})".format(lhs,  DELTA, formatSum(boundary))

            boundary_coproducts = Counter()
            if boundary is None:
                boundary_coproducts = None
            else:
                for k, v in boundary.items():
                    boundary_coproducts += Counter({l: w * v for l, w in result["coproducts"][k].items()})
                boundary_coproducts = reduceModN(boundary_coproducts)
            print " " * len(lhs) + " = " + formatSum(boundary_coproducts) + "\n"

            # Compute and print (1 \otimes \partial + \partial \otimes 1) \Delta (X)

            coproduct = result["coproducts"][i]
            lhs = CHAINPARTIAL + DELTA + str(i)
            print u"{} = {} ({})".format(lhs, CHAINPARTIAL, formatSum(coproduct))

            fullDifferential = Counter()
            for (l, r), v in coproduct.items():
                if l in result["differentials"].keys():
                    fullDifferential += Counter({(l_diff, r): v*w for l_diff, w in result["differentials"][l].items()})
                if r in result["differentials"].keys():
                    fullDifferential += Counter({(l, r_diff): v*w for r_diff, w in result["differentials"][r].items()})
            fullDifferential = reduceModN(fullDifferential)
            if not fullDifferential:
                fullDifferential = None
            print u"{} = {}".format(" "*len(lhs), formatSum(fullDifferential)) + "\n"

            # Compare two results
            if fullDifferential == boundary_coproducts:
                print "Diagonal Valid!: " + DELTA + PARTIAL + str(i) + " == " + CHAINPARTIAL + DELTA + str(i)
            else:
                allValid = False
                print "Diagonal Invalid!: " + DELTA + PARTIAL + str(i) + " != " + CHAINPARTIAL + DELTA + str(i)

    print "All Diagonals Valid!" if allValid else "Invalid Diagonal Detected!"
