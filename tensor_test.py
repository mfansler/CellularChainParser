import sys
import CellChainParse
from Coalgebra import Coalgebra
from re import sub

__author__ = 'mfansler'

def format_cells(cells):
    return sub(',', '_', sub(r'[{}]', '', str(cells)))


# Check if input file is specified
if len(sys.argv) == 2:
    f = open(sys.argv[1])
    data = f.read()
    f.close()
    result = CellChainParse.parse(data)

    if not result:
        raise SystemExit

    C = Coalgebra(result["groups"], result["differentials"], result["coproducts"])

    print C * C