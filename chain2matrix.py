__author__ = 'mfansler'

import sys
from re import sub
import CellChainParse


def format_cells(cells):
    return sub(r'[{}]', '', str(cells))


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

    for n, entries in differential.iteritems():

        print "d{} = matrix(Integers(2), {}, {}, {}, sparse=True)".format(
            n, len(result["groups"][n-1]), len(result["groups"][n]), entries
        )

    print "var({})".format(", ".join(["'" + format_cells(cell) + "'" for group in result["groups"].values() for cell in group]))

    first = True
    print "{ ",
    for n, cells in result["groups"].items():
        if not first:
            print ", ",
        else:
            first = False
        print "{}: matrix(SR, {})".format(topDimension - n, format_cells(cells)),
    print "}"

