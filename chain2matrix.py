__author__ = 'mfansler'

import sys

import CellChainParse

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

    print result["coproducts"]

