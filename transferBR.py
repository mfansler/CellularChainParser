# Standard imports
from argparse import ArgumentParser, FileType
from codecs import getwriter
from os import remove as rm
from re import compile
from subprocess import check_output, CalledProcessError
import sys

# Local imports
import CellChainParse
from support_functions import *
from formatting import *
from Coalgebra import Coalgebra

__author__ = 'mfansler'
TEMP_MAT = "~transfer-temp.mat"


def main():
    # Needed to output utf-8
    sys.stdout = getwriter('utf-8')(sys.stdout)

    # Set up commandline argument parsing
    argparser = ArgumentParser(description="Computes induced coproduct on homology")
    argparser.add_argument('--hgroups', '-hg', dest='homology_groups',
                           type=FileType('r'), help="File containing homology groups")
    argparser.add_argument('file', type=file, help="LaTeX file to be parsed")

    # Read the arguments (if available)
    try:
        args = argparser.parse_args()
    except Exception as e:
        print e.message
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

    def delta_c_chain(chain):
        return list_mod(reduce(lambda acc, cell: acc + delta_c[cell], chain, []))

    """
    Check Coassociativity on Delta_C
    """

    # (1 x Delta) Delta and  (Delta x 1) Delta
    id_x_Delta_Delta = {}
    Delta_x_id_Delta = {}
    for c, cxc in delta_c.iteritems():
        id_x_Delta_Delta[c] = [(l,) + r_cp for (l, r) in cxc for r_cp in delta_c[r]]
        Delta_x_id_Delta[c] = [l_cp + (r,) for (l, r) in cxc for l_cp in delta_c[l]]

    # DeltaC = (1 x Delta_C + Delta_C x 1) Delta_C
    id_x_Delta_Delta_x_id_Delta = add_maps_mod_2(id_x_Delta_Delta, Delta_x_id_Delta)
    print DELTA + "_c is co-associative?", not any(id_x_Delta_Delta_x_id_Delta.values())

    if any(id_x_Delta_Delta_x_id_Delta.values()):
        print u"\n(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + " =",
        print format_morphism(id_x_Delta_Delta_x_id_Delta)
        print u"\n(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + " (factored) =",
        print format_morphism({cell: factorize(factorize_cycles(chain, C), C) for cell, chain in id_x_Delta_Delta_x_id_Delta.iteritems()})

    """
    COMPUTE HOMOLOGY
    """

    if not args.homology_groups:

        # create temporary file for CHomP
        scratch = open(TEMP_MAT, 'w+')

        differential = {n: C.incidence_matrix(n, sparse=False) for n in range(1, C.topDimension() + 1)}

        def compare_incidences(x, y):
            return x[1] - y[1] if x[1] != y[1] else x[0] - y[0]

        for n, entries in differential.iteritems():
            print >> scratch, n - 1
            incidences = [(l, r, v % 2) for (l, r), v in entries.iteritems()]
            for entry in ["{} {} {}".format(l, r, v) for (l, r, v) in sorted(incidences, cmp=compare_incidences)]:
                print >> scratch, entry

        scratch.close()

        try:
            chomp_results = check_output(["chomp-matrix", TEMP_MAT, "-g"])
            #print chomp_results
        except CalledProcessError as e:
            print e.returncode
            print e.output
            print e.output
        finally:
            rm(TEMP_MAT)  # clean up

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

    print
    print "g = ", format_morphism(g)


    def g_tensor(tp):
        return tuple(map(lambda c: g[c], list(tp)))


    # generate f: C -> H
    f, integrate = generate_f_integral(C, g)


    def f_tensor(tp):
        return tuple(map(f, list(tp)))


    """
    COMPUTE Delta_2, g^2
    """

    # define Delta g
    delta_g = {h: delta_c_chain(cycle) for h, cycle in g.iteritems()}
    print
    print DELTA + u"g =", format_morphism(delta_g)

    # print
    # print DELTA + u"g (unsimplified) =", delta_g

    factored_delta_g = {cell: factorize_cycles(chain, C) for cell, chain in delta_g.iteritems()}
    print
    print DELTA + u"g (factored) =", format_morphism(factored_delta_g)

    delta2 = {}
    for cell, chain in factored_delta_g.iteritems():
        delta2[cell] = [f_tensor(cxc) for cxc in chain]

    # flatten delta2 and remove any empty elements
    delta2 = chain_map_mod(expand_map_all(delta2))
    print
    print DELTA + u"_2 =", format_morphism(delta2)

    # however, we still need all keys to be available
    for h in g.iterkeys():
        if h not in delta2:
            delta2[h] = []

    # (g x g) Delta2
    gxgDelta = {}
    for cell, chain in delta2.iteritems():
        gxgDelta[cell] = [g_tensor(cxc) for cxc in chain]
    gxgDelta = chain_map_mod(expand_map_all(gxgDelta))
    print
    print u"(g " + OTIMES + " g)" + DELTA + "^2 =", format_morphism(gxgDelta)

    # nabla g^2
    nabla_g2 = add_maps_mod_2(gxgDelta, delta_g)
    print
    print u"(g " + OTIMES + " g)" + DELTA + "^2 + " + DELTA + "g =", format_morphism(nabla_g2)

    # g^2
    g2 = {cell: integrate(chain) for cell, chain in nabla_g2.iteritems()}
    print
    print u"g^2 =", format_morphism(g2)

    """
    VERIFY CONSISTENCY OF DELTA2, g^2 RESULTS
    """

    nabla_g2_computed = {cell: derivative(chain, C) for cell, chain in g2.iteritems() if chain}
    nabla_g2_computed = chain_map_mod(expand_map_all(nabla_g2_computed))
    print
    print NABLA + u" g^2 =", format_morphism(nabla_g2_computed)

    print u"\n(g " + OTIMES + " g)" + DELTA + "^2 + " + DELTA + "g + " + NABLA + "g^2 = 0 ? ",
    print not any(add_maps_mod_2(nabla_g2_computed, nabla_g2).values())

    """
    VERIFY DELTA2 COASSOCIATIVITY
    """

    print "\nChecking coassociativity...\n"

    # (1 x Delta_2) Delta_2
    # (Delta_2 x 1) Delta_2
    id_x_Delta2_Delta2 = {}
    Delta2_x_id_Delta2 = {}
    for h, hxhs in delta2.iteritems():
        id_x_Delta2_Delta2[h] = [(l,) + r_cp for (l, r) in hxhs for r_cp in delta2[r]]
        Delta2_x_id_Delta2[h] = [l_cp + (r,) for (l, r) in hxhs for l_cp in delta2[l]]
    id_x_Delta2_Delta2 = chain_map_mod(expand_map_all(id_x_Delta2_Delta2))
    Delta2_x_id_Delta2 = chain_map_mod(expand_map_all(Delta2_x_id_Delta2))

    print u"\n(1 " + OTIMES + " " + DELTA + "^2) " + DELTA + "^2 =", format_morphism(id_x_Delta2_Delta2)
    print u"\n(" + DELTA + "^2 " + OTIMES + " 1) " + DELTA + "^2 =", format_morphism(Delta2_x_id_Delta2)

    # z_1 = (1 x Delta_2 + Delta_2 x 1) Delta_2
    z1 = add_maps_mod_2(id_x_Delta2_Delta2, Delta2_x_id_Delta2)

    print u"\n" + DELTA + "^2 is co-associative?", not any(z1.values())
    if any(z1.values()):
        print "\nz1 = " + NABLA + "(" + DELTA + "^3) =",
        print format_morphism(z1)
        print "\nz1 = " + NABLA + "(" + DELTA + "^3) (factored) =",
        print format_morphism({cell: factorize(factorize_cycles(chain, C), C) for cell, chain in z1.iteritems()})

    """
    COMPUTE DELTA_C_3
    """

    # Delta_c3
    delta_c3 = integrate(id_x_Delta_Delta_x_id_Delta)
    delta_c3 = chain_map_mod(expand_map_all(delta_c3))
    print
    print DELTA + u"_C3 =", format_morphism(delta_c3)

    # however, we still need all keys to be available
    for cell in delta_c.iterkeys():
        if cell not in delta_c3:
            delta_c3[cell] = []

    # verify consistency
    nabla_delta_c3_computed = derivative(delta_c3, C)
    nabla_delta_c3_computed = chain_map_mod(expand_map_all(nabla_delta_c3_computed))

    print
    print NABLA + DELTA + u"_C3 =", format_morphism(nabla_delta_c3_computed)

    print u"\n(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + " + " + NABLA + DELTA + u"_C3 = 0 ? ",
    print not any(add_maps_mod_2(id_x_Delta_Delta_x_id_Delta, nabla_delta_c3_computed).values())

    if any(add_maps_mod_2(id_x_Delta_Delta_x_id_Delta, nabla_delta_c3_computed).values()):
        print u"\n(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + " + " + NABLA + DELTA + u"_C3 =",
        print format_morphism(add_maps_mod_2(id_x_Delta_Delta_x_id_Delta, nabla_delta_c3_computed))

    """
    COMPUTE DELTA3, g^3
    """

    # (1 x Delta) g^2 and (Delta x 1) g^2
    id_x_Delta_g2 = {}
    Delta_x_id_g2 = {}
    for h, cxcs in g2.iteritems():
        id_x_Delta_g2[h] = [(l,) + r_cp for (l, r) in cxcs for r_cp in delta_c[r]]
        Delta_x_id_g2[h] = [l_cp + (r,) for (l, r) in cxcs for l_cp in delta_c[l]]
    id_x_Delta_g2 = chain_map_mod(id_x_Delta_g2)
    Delta_x_id_g2 = chain_map_mod(Delta_x_id_g2)
    print u"\n(1 " + OTIMES + " " + DELTA + ") g^2 =",  format_morphism(id_x_Delta_g2)
    print u"\n(" + DELTA + " " + OTIMES + " 1) g^2 =",  format_morphism(Delta_x_id_g2)

    # (g x g^2) Delta_2 and (g^2 x g) Delta_2
    g_x_g2_Delta2 = {}
    g2_x_g_Delta2 = {}
    for h, hxhs in delta2.iteritems():
        g_x_g2_Delta2[h] = [(l_cp,) + r_cp for l, r in hxhs for l_cp in g[l] for r_cp in g2[r]]
        g2_x_g_Delta2[h] = [l_cp + (r_cp,) for l, r in hxhs for l_cp in g2[l] for r_cp in g[r]]
    g_x_g2_Delta2 = chain_map_mod(g_x_g2_Delta2)
    g2_x_g_Delta2 = chain_map_mod(g2_x_g_Delta2)
    print u"\n( g " + OTIMES + " g^2 ) " + DELTA + "^2 =",  format_morphism(g_x_g2_Delta2)
    print u"\n( g^2 " + OTIMES + " g ) " + DELTA + "^2 =",  format_morphism(g2_x_g_Delta2)

    # delta_c3_g
    delta_c3_g = {}
    for h, chain in g.iteritems():
        delta_c3_g[h] = []
        for cell in chain:
            if cell in delta_c3:
                delta_c3_g[h] += delta_c3[cell]
    delta_c3_g = chain_map_mod(delta_c3_g)
    print u"\n" + DELTA + "_c3 g =",  format_morphism(delta_c3_g)

    # phi_1 = (1 x Delta + Delta x 1) g^2 + (g x g^2 + g^2 x g) Delta_2
    phi_1 = reduce(add_maps_mod_2, [g_x_g2_Delta2, g2_x_g_Delta2, id_x_Delta_g2, Delta_x_id_g2, delta_c3_g], {})
    print u"\n" + PHI + u"_1 = (g " + OTIMES + " g^2 + g^2 " + OTIMES + " g) " + DELTA + "^2 +",
    print u"(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) g^2 + " + DELTA + "_c3 g =",
    print format_morphism(phi_1)

    # Nabla phi_1 == 0 ? (Verify consistency)
    nabla_phi_1 = {}
    for h, chain in phi_1.iteritems():
        nabla_phi_1[h] = derivative(chain, C)
    nabla_phi_1 = chain_map_mod(expand_map_all(nabla_phi_1))
    print "\n" + NABLA + PHI + u"_1 =", format_morphism(nabla_phi_1)

    # factor phi_1
    factored_phi_1 = {h: factorize_cycles(chain, C) for h, chain in phi_1.iteritems() if chain}
    print "\n" + PHI + u"_1 (factored) =", format_morphism(factored_phi_1)

    # delta3 = {h: [f_tensor(cxcxc) for cxcxc in cycles] for h, cycles in factored_phi_1.iteritems()}
    # manually define delta3
    delta3 = {
        'h2_1': [('h1_0', 'h1_1', 'h1_2'), ('h1_1', 'h1_0', 'h1_2'),
                 ('h1_2', 'h1_1', 'h1_0'), ('h1_2', 'h1_0', 'h1_1')],
        'h2_0': [('h1_0', 'h1_2', 'h1_1'), ('h1_2', 'h1_1', 'h1_0'),
                 ('h1_0', 'h1_1', 'h1_2'), ('h1_1', 'h1_2', 'h1_0')]}

    # flatten delta3 and remove any empty elements (mod 2)
    delta3 = chain_map_mod(expand_map_all(delta3))
    print "\n" + DELTA + u"_3 =", format_morphism(delta3)

    # however, we still need all keys to be available
    for h in g.iterkeys():
        if h not in delta3:
            delta3[h] = []

    """
    Check coassociativity of Delta^3
    """

    print "\nChecking coassociativity...\n"

    # (Delta_3 x 1 + 1 x Delta_3) Delta_2
    delta3_x_id_delta2 = {}
    id_x_delta3_delta2 = {}

    for h, hxhs in delta2.iteritems():
        delta3_x_id_delta2[h] = [l_cp + (r, ) for (l, r) in hxhs for l_cp in delta3[l]]
        id_x_delta3_delta2[h] = [(l, ) + r_cp for (l, r) in hxhs for r_cp in delta3[r]]

    delta3_x_id_delta2 = chain_map_mod(expand_map_all(delta3_x_id_delta2))
    id_x_delta3_delta2 = chain_map_mod(expand_map_all(id_x_delta3_delta2))

    print u"\n( " + DELTA + "^3 " + OTIMES + " 1 ) " + DELTA + "^2 =",  format_morphism(delta3_x_id_delta2)
    print u"\n( 1 " + OTIMES, DELTA + "^3 ) " + DELTA + "^2 =",  format_morphism(id_x_delta3_delta2)

    # (1 x 1 x Delta_c2) Delta_c3 ## (1 x 1 x Delta_c2) Delta_c3 ## (1 x 1 x Delta_c2) Delta_c3 #
    id_x_id_x_delta2_delta3 = {}
    id_x_delta2_x_id_delta3 = {}
    delta2_x_id_x_id_delta3 = {}

    for h, hxhxhs in delta3.iteritems():
        id_x_id_x_delta2_delta3[h] = [(l, m) + r_cp for (l, m, r) in hxhxhs for r_cp in delta2[r]]
        id_x_delta2_x_id_delta3[h] = [(l, ) + m_cp + (r, ) for (l, m, r) in hxhxhs for m_cp in delta2[m]]
        delta2_x_id_x_id_delta3[h] = [l_cp + (m, r) for (l, m, r) in hxhxhs for l_cp in delta2[l]]

    id_x_id_x_delta2_delta3 = chain_map_mod(expand_map_all(id_x_id_x_delta2_delta3))
    id_x_delta2_x_id_delta3 = chain_map_mod(expand_map_all(id_x_delta2_x_id_delta3))
    delta2_x_id_x_id_delta3 = chain_map_mod(expand_map_all(delta2_x_id_x_id_delta3))

    print u"\n( 1 " + OTIMES + " 1 " + OTIMES, DELTA + "^2 ) " + DELTA + "^3 =",  format_morphism(id_x_id_x_delta2_delta3)
    print u"\n( 1 " + OTIMES, DELTA + "^2 " + OTIMES + " 1 ) " + DELTA + "^3 =",  format_morphism(id_x_delta2_x_id_delta3)
    print u"\n( " + DELTA + "^2 " + OTIMES + " 1 " + OTIMES + " 1 ) " + DELTA + "^3 =",  format_morphism(delta2_x_id_x_id_delta3)

    z2 = reduce(add_maps_mod_2, [
        delta3_x_id_delta2, id_x_delta3_delta2,
        id_x_id_x_delta2_delta3, id_x_delta2_x_id_delta3, delta2_x_id_x_id_delta3], {})

    # DeltaC = (1 x Delta_C + Delta_C x 1) Delta_C
    print "\n" + DELTA + "^3 is co-associative?", not any(z2.values())

    if any(z2.values()):
        print "\n" + NABLA + "(" + DELTA + "^4) =",
        print format_morphism(z2)
        print "\n" + NABLA + "(" + DELTA + "^4) (factored) =",
        print format_morphism({cell: factorize(factorize_cycles(chain, C), C) for cell, chain in z2.iteritems()})

    """
    Compute g^3
    """

    # (g x g x g) Delta3
    gxgxg_delta3 = {}
    for h, chain in delta3.iteritems():
        gxgxg_delta3[h] = [g_tensor(hxhxh) for hxhxh in chain]
    gxgxg_delta3 = chain_map_mod(expand_map_all(gxgxg_delta3))
    print u"\n(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "^3 =", format_morphism(gxgxg_delta3)

    # nabla g^3
    nabla_g3 = add_maps_mod_2(gxgxg_delta3, phi_1)
    print u"\n(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "^3 + " + PHI + "_1 =", format_morphism(nabla_g3)

    # g^3
    # g3 = {h: chain_integrate(chain, C) for h, chain in nabla_g3.iteritems()}
    # manually define g^3
    g3 = {
        'h0_0': [],
        'h1_0': [
            (['m_{11}', 'm_{4}'], ['m_{4}'], ['m_{4}'])],
        'h1_1': [
            (['m_{4}'], ['c_{3}', 'c_{7}'], ['m_{4}', 'c_{3}']),
            (['c_{3}', 'c_{7}'], ['c_{3}'], ['m_{4}', 'c_{3}']),
            (['c_{3}', 'c_{7}'], ['m_{4}'], ['m_{4}'])],
        'h1_2': [
            (['m_{4}', 'c_{3}', 'm_{2}'], ['m_{6}', 'm_{9}'], ['m_{4}', 'c_{3}', 'm_{2}', 'm_{6}']),
            (['m_{4}'], ['c_{3}', 'm_{2}'], ['m_{6}', 'm_{9}']),
            (['c_{3}'], ['m_{2}'], ['m_{6}', 'm_{9}']),
            (['m_{6}', 'm_{9}'], ['m_{6}'], ['m_{6}']),
            (['m_{6}', 'm_{9}'], ['m_{2}', 'm_{6}'], ['m_{4}', 'c_{3}', 'm_{2}']),
            (['m_{6}', 'm_{9}'], ['c_{3}'], ['m_{4}', 'c_{3}']),
            (['m_{6}', 'm_{9}'], ['m_{4}'], ['m_{4}'])
        ],
        'h2_1': [
            (['m_{2}', 'c_{3}', 'm_{4}'], ['t_{7}', 't_{8}', 't_{6}', 't_{5}'], ['m_{4}', 'c_{3}', 'm_{13}', 'm_{7}']),
            (['c_{3}', 'm_{4}'], ['m_{2}', 'c_{3}'], ['t_{5}', 't_{6}', 't_{8}', 't_{7}']),
            (['c_{3}'], ['c_{3}'], ['t_{5}', 't_{6}', 't_{8}', 't_{7}']),
            (['t_{6}', 't_{5}', 't_{8}', 't_{7}'], ['m_{4}', 'c_{3}', 'm_{13}', 'm_{7}'], ['m_{4}', 'c_{3}', 'm_{13}', 'm_{7}']),
            (['t_{5}', 't_{7}', 't_{8}', 't_{6}'], ['m_{4}', 'c_{3}', 'm_{13}'], ['m_{7}']),
            (['t_{5}', 't_{8}', 't_{7}', 't_{6}'], ['m_{4}', 'c_{3}'], ['m_{13}']),
            (['t_{5}', 't_{8}', 't_{7}', 't_{6}'], ['m_{4}'], ['c_{3}']),
            (['m_{4}', 'm_{11}'], ['c_{3}', 'c_{7}'], ['t_{5}', 't_{6}']),
            (['m_{6}', 'm_{9}'], ['t_{5}', 't_{6}'], ['c_{13}', 'c_{14}']),
            (['m_{6}', 'm_{9}'], ['m_{6}', 'm_{9}'], ['s_{9}', 's_{10}']),
            (['t_{5}', 'm_{6}'], ['c_{5}', 'c_{9}'], ['m_{7}', 'm_{8}']),
            (['m_{7}', 'm_{8}'], ['s_{1}', 's_{2}', 's_{9}', 's_{10}'], ['m_{7}', 'm_{8}']),
            (['m_{4}', 'c_{3}', 'm_{2}'], ['m_{6}', 'm_{9}'], ['s_{1}', 's_{2}', 's_{5}', 's_{6}', 's_{9}', 's_{10}']),
            (['m_{6}', 'm_{9}'], ['s_{1}', 's_{2}', 's_{5}', 's_{6}', 's_{9}', 's_{10}'], ['m_{4}', 'c_{3}', 'm_{13}', 'c_{6}', 'm_{3}']),
            (['m_{9}', 'm_{6}'], ['m_{2}', 'c_{3}', 'm_{2}', 'm_{6}'], ['s_{1}', 's_{2}', 's_{9}', 's_{10}']),
            (['s_{1}', 's_{2}', 's_{5}', 's_{6}', 's_{9}', 's_{10}'], ['m_{7}', 'm_{8}'], ['m_{4}', 'c_{3}', 'm_{13}', 'm_{7}']),
            (['m_{4}', 'c_{3}', 'm_{2}', 'm_{6}', 'c_{5}'], ['s_{5}', 's_{6}'], ['m_{7}', 'm_{8}']),
            (['m_{6}', 'm_{9}'], ['c_{9}'], ['s_{5}', 's_{6}']),
            (['s_{1}', 's_{2}', 's_{5}', 's_{6}', 's_{9}', 's_{10}'], ['m_{3}', 'c_{10}'], ['m_{7}', 'm_{8}']),
            (['c_{6}', 'c_{10}'], ['s_{6}'], ['m_{8}', 'm_{7}']),
            (['m_{6}', 'm_{9}'], ['m_{4}'], ['s_{1}', 's_{2}', 's_{9}', 's_{10}']),
            (['m_{6}', 'm_{9}'], ['m_{2}'], ['s_{1}', 's_{2}', 's_{9}', 's_{10}']),
            (['m_{6}', 'm_{9}'], ['c_{3}', 'c_{7}'], ['s_{2}', 's_{10}', 'a_{1}', 'a_{3}']),
            (['c_{3}', 'c_{7}'], ['s_{2}', 's_{10}', 'a_{1}', 'a_{3}'], ['m_{7}', 'm_{8}']),
            (['c_{3}', 'c_{7}'], ['m_{4}', 'm_{11}'], ['t_{5}', 't_{6}']),
            (['m_{6}'], ['c_{5}', 'c_{9}'], ['m_{7}', 'm_{8}']),
            (['m_{6}', 'm_{9}'], ['c_{6}', 'c_{10}'], ['s_{5}']),
            (['m_{6}', 'm_{9}'], ['s_{2}', 's_{6}', 's_{10}', 'a_{1}', 'a_{3}'], ['c_{13}', 'c_{14}']),
            (['s_{1}', 's_{5}', 's_{9}', 'a_{1}', 'a_{3}'], ['c_{13}', 'c_{14}'], ['m_{7}', 'm_{8}']),
            (['t_{6}'], ['c_{5}', 'c_{9}'], ['m_{7}', 'm_{8}']),
            (['t_{5}', 't_{6}'], ['c_{6}', 'c_{10}'], ['m_{7}', 'm_{8}']),
            (['m_{7}', 'm_{8}'], ['s_{5}', 's_{6}'], ['m_{7}', 'm_{8}']),
            (['m_{6}', 'm_{9}'], ['t_{5}', 't_{6}'], ['c_{3}', 'c_{7}']),
            (['m_{6}', 'm_{9}'], ['m_{7}', 'm_{8}'], ['s_{9}', 's_{10}'])
        ],
        'h2_0': [
            (['t_{4}', 't_{2}', 't_{3}', 't_{1}'], ['m_{1}', 'm_{3}', 'm_{12}'], ['m_{1}']),
            (['t_{4}', 't_{2}', 't_{3}', 't_{1}'], ['m_{12}'], ['m_{3}']),
            (['s_{5}', 's_{6}', 's_{7}', 's_{8}', 's_{9}', 's_{10}'], ['m_{5}', 'm_{10}'], ['m_{1}', 'm_{3}', 'm_{12}']),
            (['m_{4}', 'm_{11}'], ['s_{5}', 's_{6}', 's_{9}', 's_{10}'], ['m_{1}']),
            (['s_{7}', 's_{8}'], ['m_{10}', 'm_{12}', 'c_{6}'], ['m_{5}', 'm_{10}']),
            (['m_{4}', 'c_{3}'], ['s_{7}', 's_{8}'], ['m_{5}', 'm_{10}']),
            (['m_{4}', 'm_{11}'], ['m_{4}'], ['s_{9}', 's_{10}']),
            (['m_{4}', 'm_{11}'], ['c_{3}', 'c_{4}', 'm_{10}', 'm_{12}'], ['s_{5}', 's_{6}']),
            (['m_{4}', 'm_{11}'], ['c_{3}'], ['s_{7}', 's_{8}']),
            (['s_{5}', 's_{6}', 's_{9}', 's_{10}'], ['c_{8}', 'm_{13}', 'm_{8}', 'c_{14}'], ['m_{5}', 'm_{10}']),
            (['m_{4}', 'c_{3}', 'm_{13}', 'c_{6}'], ['s_{5}', 's_{6}'], ['m_{5}', 'm_{10}']),
            (['m_{4}', 'm_{11}'], ['s_{7}', 's_{8}'], ['m_{1}', 'c_{13}', 'm_{7}']),
            (['m_{4}', 'm_{11}'], ['m_{4}', 'm_{11}'], ['s_{9}', 's_{10}']),
            (['m_{4}', 'm_{11}'], ['t_{5}', 't_{6}'], ['c_{7}', 'c_{3}']),
            (['m_{4}', 'm_{11}'], ['m_{8}', 'm_{7}'], ['s_{9}', 's_{10}']),
            (['t_{5}', 't_{6}'], ['c_{3}', 'c_{7}'], ['m_{4}', 'm_{11}']),
            (['m_{8}', 'm_{7}'], ['s_{9}', 's_{10}'], ['m_{4}', 'm_{11}']),
            (['m_{8}', 'm_{7}'], ['c_{14}', 'c_{13}'], ['t_{1}', 't_{2}']),
            (['m_{4}', 'm_{11}'], ['s_{8}'], ['c_{13}', 'c_{14}']),
            (['m_{4}', 'm_{11}'], ['c_{4}', 'c_{8}'], ['s_{6}']),
            (['s_{8}'], ['c_{6}', 'c_{10}'], ['m_{5}', 'm_{10}']),
            (['c_{6}', 'c_{10}'], ['s_{8}'], ['m_{5}', 'm_{10}']),
            (['m_{4}', 'm_{11}'], ['s_{9}', 's_{10}'], ['m_{5}', 'm_{10}']),
            (['m_{4}', 'm_{11}'], ['c_{3}', 'c_{7}'], ['s_{6}', 's_{8}']),
            (['m_{4}', 'm_{11}'], ['c_{7}', 'c_{3}'], ['t_{5}', 't_{6}']),
            (['s_{5}', 's_{6}', 's_{9}', 's_{10}'], ['m_{6}', 'm_{9}'], ['m_{4}', 'm_{11}']),
            (['c_{6}', 'c_{10}'], ['t_{5}', 't_{6}'], ['m_{4}', 'm_{11}']),
            (['c_{10}', 'c_{6}'], ['m_{7}', 'm_{8}'], ['t_{1}', 't_{2}'])
        ]
    }

    g3 = chain_map_mod(expand_map_all(g3))
    print u"\ng^3 =", format_morphism(g3)

    # however, we still need all keys to be available
    for h in g.iterkeys():
        if h not in g3:
            g3[h] = []

    """
    VERIFY CONSISTENCY OF phi_1, Delta^3, and g^3
    """

    nabla_g3_computed = {h: derivative(chain, C) for h, chain in g3.iteritems() if chain}
    nabla_g3_computed = chain_map_mod(expand_map_all(nabla_g3_computed))
    print "\n" + NABLA + u" g^3 =", format_morphism(nabla_g3_computed)

    print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "^3 + " + PHI + "_1 + " + NABLA + "g^3 = 0 ? ",
    print not any(reduce(add_maps_mod_2, [gxgxg_delta3, nabla_g3_computed, phi_1], {}).values())

    if any(reduce(add_maps_mod_2, [gxgxg_delta3, nabla_g3_computed, phi_1], {}).values()):
        print "\t", reduce(add_maps_mod_2, [gxgxg_delta3, nabla_g3_computed, phi_1], {})

    """
    COMPUTE \Delta_C4
    """

    print "\n\nComputing", DELTA + u"_C4...\n\n"

    # (Delta_c3 x 1 + 1 x Delta_c3) Delta_c2
    delta_c3_x_id_delta_c = {}
    id_x_delta_c3_delta_c = {}

    for cell, cxcs in delta_c.iteritems():
        delta_c3_x_id_delta_c[cell] = [l_cp + (r, ) for (l, r) in cxcs for l_cp in delta_c3[l]]
        id_x_delta_c3_delta_c[cell] = [(l, ) + r_cp for (l, r) in cxcs for r_cp in delta_c3[r]]

    delta_c3_x_id_delta_c = chain_map_mod(expand_map_all(delta_c3_x_id_delta_c))
    id_x_delta_c3_delta_c = chain_map_mod(expand_map_all(id_x_delta_c3_delta_c))

    print u"\n( " + DELTA + "_c3 " + OTIMES + " 1 ) " + DELTA + "_c2 =",  format_morphism(delta_c3_x_id_delta_c)
    print u"\n( 1 " + OTIMES, DELTA + "_c3 ) " + DELTA + "_c2 =",  format_morphism(id_x_delta_c3_delta_c)

    # (1 x 1 x Delta_c2) Delta_c3 ## (1 x 1 x Delta_c2) Delta_c3 ## (1 x 1 x Delta_c2) Delta_c3 #
    id_x_id_x_delta_c2_delta_c3 = {}
    id_x_delta_c2_x_id_delta_c3 = {}
    delta_c2_x_id_x_id_delta_c3 = {}

    for cell, cxcxcs in delta_c3.iteritems():
        id_x_id_x_delta_c2_delta_c3[cell] = [(l, m) + r_cp for (l, m, r) in cxcxcs for r_cp in delta_c[r]]
        id_x_delta_c2_x_id_delta_c3[cell] = [(l, ) + m_cp + (r, ) for (l, m, r) in cxcxcs for m_cp in delta_c[m]]
        delta_c2_x_id_x_id_delta_c3[cell] = [l_cp + (m, r) for (l, m, r) in cxcxcs for l_cp in delta_c[l]]

    id_x_id_x_delta_c2_delta_c3 = chain_map_mod(expand_map_all(id_x_id_x_delta_c2_delta_c3))
    id_x_delta_c2_x_id_delta_c3 = chain_map_mod(expand_map_all(id_x_delta_c2_x_id_delta_c3))
    delta_c2_x_id_x_id_delta_c3 = chain_map_mod(expand_map_all(delta_c2_x_id_x_id_delta_c3))

    print u"\n( 1 " + OTIMES + " 1 " + OTIMES, DELTA + "_c2 ) " + DELTA + "_c3 =",  format_morphism(id_x_id_x_delta_c2_delta_c3)
    print u"\n( 1 " + OTIMES, DELTA + "_c2 " + OTIMES + " 1 ) " + DELTA + "_c3 =",  format_morphism(id_x_delta_c2_x_id_delta_c3)
    print u"\n( " + DELTA + "_c2 " + OTIMES + " 1 " + OTIMES + " 1 ) " + DELTA + "_c3 =",  format_morphism(delta_c2_x_id_x_id_delta_c3)

    nabla_delta_c4 = reduce(add_maps_mod_2, [
        delta_c3_x_id_delta_c, id_x_delta_c3_delta_c,
        id_x_id_x_delta_c2_delta_c3, id_x_delta_c2_x_id_delta_c3, delta_c2_x_id_x_id_delta_c3], {})

    # DeltaC = (1 x Delta_C + Delta_C x 1) Delta_C
    print DELTA + "_c3 is co-associative?", not any(nabla_delta_c4.values())

    if any(nabla_delta_c4.values()):
        print "\n" + NABLA + "(" + DELTA + "_C4) =",
        print format_morphism(nabla_delta_c4)
        print "\n" + NABLA + "(" + DELTA + "_C4) (factored) =",
        print format_morphism({cell: factorize(factorize_cycles(chain, C), C) for cell, chain in nabla_delta_c4.iteritems()})

    # Delta_c4
    delta_c4 = integrate(nabla_delta_c4)
    delta_c4 = chain_map_mod(expand_map_all(delta_c4))
    print
    print DELTA + u"_C4 =", format_morphism(delta_c4)

    # verify consistency
    nabla_delta_c4_computed = derivative(delta_c4, C)
    nabla_delta_c4_computed = chain_map_mod(expand_map_all(nabla_delta_c4_computed))

    print
    print NABLA + DELTA + u"_C4 =", format_morphism(nabla_delta_c4_computed)

    print u"\n(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + " + " + NABLA + DELTA + u"_C3 = 0 ? ",
    print not any(add_maps_mod_2(nabla_delta_c4, nabla_delta_c4_computed).values())

    if any(add_maps_mod_2(nabla_delta_c4, nabla_delta_c4_computed).values()):
        print u"\n(1 " + OTIMES + " " + DELTA + " + " + DELTA + " " + OTIMES + " 1) " + DELTA + " + " + NABLA + DELTA + u"_C3 =",
        print format_morphism(add_maps_mod_2(nabla_delta_c4, nabla_delta_c4_computed))

    """
    COMPUTE \Phi_2, \Delta4
    """

    #####################
    # Facets of J_4
    #####################

    # (Delta_C4) g #
    delta_c4_g = {}
    for h, chain in g.iteritems():
        delta_c4_g[h] = []
        for cell in chain:
            if cell in delta_c4:
                delta_c4_g[h] += delta_c4[cell]
    delta_c4_g = chain_map_mod(delta_c4_g)
    print u"\n" + DELTA + "_c4 g =",  format_morphism(delta_c4_g)

    # (1 x Delta_C3) g^2 ## (Delta_C3 x 1) g^2 #
    id_x_Delta_c3_g2 = {}
    Delta_c3_x_id_g2 = {}
    for h, cxcs in g2.iteritems():
        id_x_Delta_c3_g2[h] = [(l, ) + r_cp for (l, r) in cxcs for r_cp in delta_c3[r]]
        Delta_c3_x_id_g2[h] = [l_cp + (r, ) for (l, r) in cxcs for l_cp in delta_c3[l]]
    id_x_Delta_c3_g2 = chain_map_mod(id_x_Delta_c3_g2)
    Delta_c3_x_id_g2 = chain_map_mod(Delta_c3_x_id_g2)

    print u"\n(1 " + OTIMES + " " + DELTA + "_C3) g^2 =",  format_morphism(id_x_Delta_c3_g2)
    print u"\n( " + DELTA + "_C3 " + OTIMES + " 1) g^2 =",  format_morphism(Delta_c3_x_id_g2)

    # (1 x 1 x Delta) g^3 ## (1 x Delta x 1) g^3 ## (Delta x 1 x 1) g^3 #
    id_x_id_x_Delta_g3 = {}
    id_x_Delta_x_id_g3 = {}
    Delta_x_id_x_id_g3 = {}
    for h, cxcxcs in g3.iteritems():
        id_x_id_x_Delta_g3[h] = [(l, m) + r_cp for (l, m, r) in cxcxcs for r_cp in delta_c[r]]
        id_x_Delta_x_id_g3[h] = [(l, ) + m_cp + (r, ) for (l, m, r) in cxcxcs for m_cp in delta_c[m]]
        Delta_x_id_x_id_g3[h] = [l_cp + (m, r) for (l, m, r) in cxcxcs for l_cp in delta_c[l]]
    id_x_id_x_Delta_g3 = chain_map_mod(id_x_id_x_Delta_g3)
    id_x_Delta_x_id_g3 = chain_map_mod(id_x_Delta_x_id_g3)
    Delta_x_id_x_id_g3 = chain_map_mod(Delta_x_id_x_id_g3)

    print u"\n(1 " + OTIMES + " 1 " + OTIMES + " " + DELTA + ") g^3 =",  format_morphism(id_x_id_x_Delta_g3)
    print u"\n(1 " + OTIMES + " " + DELTA + " " + OTIMES + " 1) g^3 =",  format_morphism(id_x_Delta_x_id_g3)
    print u"\n(" + DELTA + " " + OTIMES + " 1 " + OTIMES + " 1) g^3 =",  format_morphism(Delta_x_id_x_id_g3)

    # (g x g^3) Delta_2 ## (g^2 x g^2) Delta_2  ## (g^3 x g) Delta_2 #
    g_x_g3_Delta2 = {}
    g2_x_g2_Delta2 = {}
    g3_x_g_Delta2 = {}

    for h, hxhs in delta2.iteritems():
        g_x_g3_Delta2[h] = [(l_cp, ) + r_cp for l, r in hxhs for l_cp in g[l] for r_cp in g3[r]]
        g2_x_g2_Delta2[h] = [l_cp + r_cp for l, r in hxhs for l_cp in g2[l] for r_cp in g2[r]]
        g3_x_g_Delta2[h] = [l_cp + (r_cp, ) for l, r in hxhs for l_cp in g3[l] for r_cp in g[r]]

    g_x_g3_Delta2 = chain_map_mod(g_x_g3_Delta2)
    g2_x_g2_Delta2 = chain_map_mod(g2_x_g2_Delta2)
    g3_x_g_Delta2 = chain_map_mod(g3_x_g_Delta2)

    print u"\n( g " + OTIMES + " g^3 ) " + DELTA + "^2 =",  format_morphism(g_x_g3_Delta2)
    print u"\n( g^2 " + OTIMES + " g^2 ) " + DELTA + "^2 =",  format_morphism(g2_x_g2_Delta2)
    print u"\n( g^3 " + OTIMES + " g ) " + DELTA + "^2 =",  format_morphism(g3_x_g_Delta2)

    # (g x g x g^2) Delta^3 ## (g x g^2 x g) Delta^3 ## (g^2 x g x g) Delta^3 #
    g_x_g_x_g2_Delta3 = {}
    g_x_g2_x_g_Delta3 = {}
    g2_x_g_x_g_Delta3 = {}

    for h, hxhxhs in delta3.iteritems():
        g_x_g_x_g2_Delta3[h] = [(g[l], g[m]) + r_cp for (l, m, r) in hxhxhs for r_cp in g2[r]]
        g_x_g2_x_g_Delta3[h] = [(g[l], ) + m_cp + (g[r], ) for (l, m, r) in hxhxhs for m_cp in g2[m]]
        g2_x_g_x_g_Delta3[h] = [l_cp + (g[m], g[r]) for (l, m, r) in hxhxhs for l_cp in g2[l]]

    g_x_g_x_g2_Delta3 = chain_map_mod(expand_map_all(g_x_g_x_g2_Delta3))
    g_x_g2_x_g_Delta3 = chain_map_mod(expand_map_all(g_x_g2_x_g_Delta3))
    g2_x_g_x_g_Delta3 = chain_map_mod(expand_map_all(g2_x_g_x_g_Delta3))

    print u"\n( g " + OTIMES + " g " + OTIMES + " g^2 ) " + DELTA + "^3 =",  format_morphism(g_x_g_x_g2_Delta3)
    print u"\n( g " + OTIMES + " g^2 " + OTIMES + " g ) " + DELTA + "^3 =",  format_morphism(g_x_g2_x_g_Delta3)
    print u"\n( g^2 " + OTIMES + " g " + OTIMES + " g ) " + DELTA + "^3 =",  format_morphism(g2_x_g_x_g_Delta3)

    # phi_2
    phi_2 = reduce(add_maps_mod_2, [delta_c4_g, id_x_Delta_c3_g2, Delta_c3_x_id_g2,
                                    id_x_id_x_Delta_g3, id_x_Delta_x_id_g3, Delta_x_id_x_id_g3,
                                    g_x_g3_Delta2, g3_x_g_Delta2, g2_x_g2_Delta2,
                                    g_x_g_x_g2_Delta3, g_x_g2_x_g_Delta3, g2_x_g_x_g_Delta3], {})

    print "\n" + PHI + u"_2 =",  format_morphism(phi_2)

    # factor phi_2
    factored_phi_2 = {h: factorize_cycles(chain, C) for h, chain in phi_2.iteritems() if chain}
    print "\n" + PHI + u"_2 (factored) =", format_morphism(factored_phi_2)

    delta4 = {h: [f_tensor(cxcxcxc) for cxcxcxc in cycles] for h, cycles in factored_phi_2.iteritems()}

    # flatten delta3 and remove any empty elements (mod 2)
    delta4 = chain_map_mod(expand_map_all(delta4))
    print "\n" + DELTA + u"_4 =", format_morphism(delta4)

    # however, we still need all keys to be available
    for h in g.iterkeys():
        if h not in delta4:
            delta4[h] = []

    """
    Check coassociativity of Delta^4
    """
    print "\nChecking coassociativity...\n"

    # (Delta^2 x 1 x 1 x 1) Delta^4 ## (1 x Delta^2 x 1 x 1) Delta^4 #
    # (1 x 1 x Delta^2 x 1) Delta^4 ## (1 x 1 x 1 x Delta^2) Delta^4 #
    delta2_x_id_x_id_x_id_Delta4 = {}
    id_x_delta2_x_id_x_id_Delta4 = {}
    id_x_id_x_delta2_x_id_Delta4 = {}
    id_x_id_x_id_x_delta2_Delta4 = {}

    for h, hxhxhxhs in delta4.iteritems():
        delta2_x_id_x_id_x_id_Delta4[h] = [cp + (h2, h3, h4) for (h1, h2, h3, h4) in hxhxhxhs for cp in delta2[h1]]
        id_x_delta2_x_id_x_id_Delta4[h] = [(h1, ) + cp + (h3, h4) for (h1, h2, h3, h4) in hxhxhxhs for cp in delta2[h2]]
        id_x_id_x_delta2_x_id_Delta4[h] = [(h1, h2) + cp + (h4, ) for (h1, h2, h3, h4) in hxhxhxhs for cp in delta2[h3]]
        id_x_id_x_id_x_delta2_Delta4[h] = [(h1, h2, h3) + cp for (h1, h2, h3, h4) in hxhxhxhs for cp in delta2[h4]]

    delta2_x_id_x_id_x_id_Delta4 = chain_map_mod(expand_map_all(delta2_x_id_x_id_x_id_Delta4))
    id_x_delta2_x_id_x_id_Delta4 = chain_map_mod(expand_map_all(id_x_delta2_x_id_x_id_Delta4))
    id_x_id_x_delta2_x_id_Delta4 = chain_map_mod(expand_map_all(id_x_id_x_delta2_x_id_Delta4))
    id_x_id_x_id_x_delta2_Delta4 = chain_map_mod(expand_map_all(id_x_id_x_id_x_delta2_Delta4))

    print u"\n( " + DELTA + "^2 " + OTIMES + " 1 " + OTIMES + " 1 " + OTIMES + " 1 ) " + DELTA + "_4 =",  format_morphism(delta2_x_id_x_id_x_id_Delta4)
    print u"\n( 1 " + OTIMES, DELTA + "^2 " + OTIMES + " 1 " + OTIMES + " 1 ) " + DELTA + "_4 =",  format_morphism(id_x_delta2_x_id_x_id_Delta4)
    print u"\n( 1 " + OTIMES + " 1 " + OTIMES, DELTA + "^2 " + OTIMES + " 1 ) " + DELTA + "_4 =",  format_morphism(id_x_id_x_delta2_x_id_Delta4)
    print u"\n( 1 " + OTIMES + " 1 " + OTIMES + " 1 " + OTIMES, DELTA + "^2 ) " + DELTA + "_4 =",  format_morphism(id_x_id_x_id_x_delta2_Delta4)

    # (Delta^3 x 1 x 1) Delta^3 ## (1 x Delta^3 x 1) Delta^3 ## (1 x 1 x Delta^3) Delta^3 #
    delta3_x_id_x_id_Delta3 = {}
    id_x_delta3_x_id_Delta3 = {}
    id_x_id_x_delta3_Delta3 = {}

    for h, hxhxhs in delta3.iteritems():
        delta3_x_id_x_id_Delta3[h] = [l_cp + (m, r) for (l, m, r) in hxhxhs for l_cp in delta3[l]]
        id_x_delta3_x_id_Delta3[h] = [(l, ) + m_cp + (r, ) for (l, m, r) in hxhxhs for m_cp in delta3[m]]
        id_x_id_x_delta3_Delta3[h] = [(l, m) + r_cp for (l, m, r) in hxhxhs for r_cp in delta3[r]]

    delta3_x_id_x_id_Delta3 = chain_map_mod(expand_map_all(delta3_x_id_x_id_Delta3))
    id_x_delta3_x_id_Delta3 = chain_map_mod(expand_map_all(id_x_delta3_x_id_Delta3))
    id_x_id_x_delta3_Delta3 = chain_map_mod(expand_map_all(id_x_id_x_delta3_Delta3))

    print u"\n( " + DELTA + "^3 " + OTIMES + " 1 " + OTIMES + " 1 ) " + DELTA + "_3 =",  format_morphism(delta3_x_id_x_id_Delta3)
    print u"\n( 1 " + OTIMES, DELTA + "^3 " + OTIMES + " 1 ) " + DELTA + "_3 =",  format_morphism(id_x_delta3_x_id_Delta3)
    print u"\n( 1 " + OTIMES + " 1 " + OTIMES, DELTA + "^3 ) " + DELTA + "_3 =",  format_morphism(id_x_id_x_delta3_Delta3)

    # (Delta^4 x 1) Delta^2 ## (1 x Delta^4) Delta^2 #
    delta4_x_id_Delta2 = {}
    id_x_delta4_Delta2 = {}

    for h, hxhs in delta2.iteritems():
        delta4_x_id_Delta2[h] = [l_cp + (r, ) for (l, r) in hxhs for l_cp in delta4[l]]
        id_x_delta4_Delta2[h] = [(l, ) + r_cp for (l, r) in hxhs for r_cp in delta4[r]]

    delta4_x_id_Delta2 = chain_map_mod(delta4_x_id_Delta2)
    id_x_delta4_Delta2 = chain_map_mod(id_x_delta4_Delta2)

    print u"\n( " + DELTA + "^4 " + OTIMES + " 1 ) " + DELTA + "_2 =",  format_morphism(delta4_x_id_Delta2)
    print u"\n( 1 " + OTIMES, DELTA + "^4 ) " + DELTA + "_2 =",  format_morphism(id_x_delta4_Delta2)

    # z3
    z3 = reduce(add_maps_mod_2, [delta2_x_id_x_id_x_id_Delta4, id_x_delta2_x_id_x_id_Delta4,
                                 id_x_id_x_delta2_x_id_Delta4, id_x_id_x_id_x_delta2_Delta4,
                                 delta3_x_id_x_id_Delta3, id_x_delta3_x_id_Delta3, id_x_id_x_delta3_Delta3,
                                 delta4_x_id_Delta2, id_x_delta4_Delta2], {})

    print DELTA + "^4 is co-associative?", not any(z3.values())

    if any(z3.values()):
        print "\nz3 = " + NABLA + "(" + DELTA + "^5) =",
        print format_morphism(z3)
        print "\nz3 = " + NABLA + "(" + DELTA + "^5) (factored) =",
        print format_morphism({cell: factorize(factorize_cycles(chain, C), C) for cell, chain in z3.iteritems()})

    """
    Compute g^4
    """

    # (g x g x g x g) Delta4
    gxgxgxg_delta4 = {}
    for h, chain in delta4.iteritems():
        gxgxgxg_delta4[h] = [g_tensor(hxhxhxh) for hxhxhxh in chain]
    gxgxgxg_delta4 = chain_map_mod(expand_map_all(gxgxgxg_delta4))
    print u"\n(g " + OTIMES + " g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_4 =", format_morphism(gxgxgxg_delta4)

    # nabla g^4
    nabla_g4 = add_maps_mod_2(gxgxgxg_delta4, phi_2)
    print u"\n(g " + OTIMES + " g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_4 + " + PHI + "_2 =", format_morphism(nabla_g4)

    # g^4
    g4 = {h: chain_integrate(chain, C) for h, chain in nabla_g4.iteritems()}
    g4 = chain_map_mod(expand_map_all(g4))
    print u"\ng^4 =", format_morphism(g4)

if __name__ == '__main__':
    main()
