# Standard imports
from sys import stdout
from re import sub
from argparse import ArgumentParser
from argparse import FileType

# Local imports
import CellChainParse
from Coalgebra import Coalgebra

__author__ = 'mfansler'


def format_cells(cells):
    return sub(',', '_', sub(r'[{}]', '', str(cells)))

argparser = ArgumentParser(description="Parses LaTeX descriptions of differential "
                                       "graded coalgebras and outputs incidence matrices")
argparser.add_argument('--sage', action='store_true', help="output sage matrix")
argparser.add_argument('--chomp', action='store_true', help="output CHomP matrix")
argparser.add_argument('--out', '-o', dest='out', type=FileType('w'), help="location to store output")
argparser.add_argument('file', type=file, help="LaTeX file to be parsed")
args = None
try:
    args = argparser.parse_args()
except Exception as e:
    print e.strerror
    argparser.print_help()
    raise SystemExit
else:
    data = args.file.read()
    args.file.close()
    result = CellChainParse.parse(data)

    if not result:
        raise SystemExit

    f = args.out if args.out else stdout

    C = Coalgebra(result["groups"], result["differentials"], result["coproducts"])

    differential = {n: C.incidence_matrix(n) for n in range(1, C.topDimension() + 1)}

    if args.sage:
        for n, entries in differential.iteritems():

            print >> f, "d{} = matrix(Integers(2), {}, {}, {}, sparse=True)".format(
                n, len(C.groups[n-1]), len(C.groups[n]), entries
            )

        print >> f, "ChainComplex({", ", ".join(["{}: d{}".format(n, n) for n in differential.keys()]), "}, degree=-1)"

        print >> f, "var({})".format(", ".join(["'" + format_cells(cell) + "'" for group in C.groups.values() for cell in group]))

        first = True
        print >> f, "{ ",
        for n, group in C.groups.items():
            if not first:
                print >> f, ", ",
            else:
                first = False
            print >> f, "{}: matrix(SR, [{}])".format(n, ", ".join(
                ["'" + format_cells(cell) + "'" for cell in group])),
        print >> f, "}"

    if args.chomp:
        for n, entries in differential.iteritems():
            print >> f, n - 1
            for entry in ["{} {} {}".format(l, r, v) for (l, r), v in entries.iteritems()]:
                print >> f, entry

    if args.out:
        f.close()
