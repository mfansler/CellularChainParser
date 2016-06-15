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

    # (g x g) Delta2
    gxgDelta = {}
    for cell, chain in delta2.iteritems():
        gxgDelta[cell] = [g_tensor(cxc) for cxc in chain]
    gxgDelta = chain_map_mod(expand_map_all(gxgDelta))
    print
    print u"(g " + OTIMES + " g)" + DELTA + "_2 =", format_morphism(gxgDelta)

    # nabla g^2
    nabla_g2 = add_maps_mod_2(gxgDelta, delta_g)
    print
    print u"(g " + OTIMES + " g)" + DELTA + "_2 + " + DELTA + "g =", format_morphism(nabla_g2)

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

    print u"\n(g " + OTIMES + " g)" + DELTA + "_2 + " + DELTA + "g + " + NABLA + "g^2 = 0 ? ",
    print not any(add_maps_mod_2(nabla_g2_computed, nabla_g2).values())

    """
    VERIFY DELTA2 COASSOCIATIVITY
    """

    # (1 x Delta_2) Delta_2
    # (Delta_2 x 1) Delta_2
    id_x_Delta2_Delta2 = {}
    Delta2_x_id_Delta2 = {}
    for h, hxhs in delta2.iteritems():
        id_x_Delta2_Delta2[h] = [(l,) + r_cp for (l, r) in hxhs for r_cp in delta2[r]]
        Delta2_x_id_Delta2[h] = [l_cp + (r,) for (l, r) in hxhs for l_cp in delta2[l]]

    # print u"\n(1 " + OTIMES + " " + DELTA + "_2) " + DELTA + "_2 =", format_morphism(id_x_Delta2_Delta2)
    # print u"\n(" + DELTA + "_2 " + OTIMES + " 1) " + DELTA + "_2 =", format_morphism(Delta2_x_id_Delta2)

    # z_1 = (1 x Delta_2 + Delta_2 x 1) Delta_2
    z_1 = add_maps_mod_2(id_x_Delta2_Delta2, Delta2_x_id_Delta2)
    # print u"\nz_1 = (1 " + OTIMES + " " + DELTA + "_2 + " + DELTA + "_2 " + OTIMES + " 1) " + DELTA + "_2 =",  format_morphism(z_1)

    print u"\n" + DELTA + "_2 is co-associative?", not any(z_1.values())

    """
    COMPUTE DELTA_C_3
    """

    # Delta_c3
    delta_c3 = integrate(id_x_Delta_Delta_x_id_Delta)
    delta_c3 = chain_map_mod(expand_map_all(delta_c3))
    print
    print DELTA + u"_C3 =", format_morphism(delta_c3)

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
    print u"\n( g " + OTIMES + " g^2 ) " + DELTA + "_2 =",  format_morphism(g_x_g2_Delta2)
    print u"\n( g^2 " + OTIMES + " g ) " + DELTA + "_2 =",  format_morphism(g2_x_g_Delta2)

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
    print u"\n" + PHI + u"_1 = (g " + OTIMES + " g^2 + g^2 " + OTIMES + " g) " + DELTA + "_2 +",
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

    delta3 = {h: [f_tensor(cxcxc) for cxcxc in cycles] for h, cycles in factored_phi_1.iteritems()}

    # flatten delta3 and remove any empty elements (mod 2)
    delta3 = chain_map_mod(expand_map_all(delta3))
    print "\n" + DELTA + u"_3 =", format_morphism(delta3)

    # (g x g x g) Delta3
    gxgxg_delta3 = {}
    for h, chain in delta3.iteritems():
        gxgxg_delta3[h] = [g_tensor(hxhxh) for hxhxh in chain]
    gxgxg_delta3 = chain_map_mod(expand_map_all(gxgxg_delta3))
    print u"\n(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 =", format_morphism(gxgxg_delta3)

    # nabla g^3
    nabla_g3 = add_maps_mod_2(gxgxg_delta3, phi_1)
    print u"\n(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 + " + PHI + "_1 =", format_morphism(nabla_g3)

    # g^3
    g3 = {h: chain_integrate(chain, C) for h, chain in nabla_g3.iteritems()}
    g3 = chain_map_mod(expand_map_all(g3))
    print u"\ng^3 =", format_morphism(g3)

    """
    VERIFY CONSISTENCY OF phi_1, Delta_3, and g^3
    """

    nabla_g3_computed = {h: derivative(chain, C) for h, chain in g3.iteritems() if chain}
    nabla_g3_computed = chain_map_mod(expand_map_all(nabla_g3_computed))
    print "\n" + NABLA + u" g^3 =", format_morphism(nabla_g3_computed)

    print u"(g " + OTIMES + " g " + OTIMES + " g)" + DELTA + "_3 + " + PHI + "_1 + " + NABLA + "g^3 = 0 ? ",
    print not any(reduce(add_maps_mod_2, [gxgxg_delta3, nabla_g3_computed, phi_1], {}).values())

    if any(reduce(add_maps_mod_2, [gxgxg_delta3, nabla_g3_computed, phi_1], {}).values()):
        print "\t", reduce(add_maps_mod_2, [gxgxg_delta3, nabla_g3_computed, phi_1], {})

    #####################
    # Facets of J_4
    #####################


if __name__ == '__main__':
    main()