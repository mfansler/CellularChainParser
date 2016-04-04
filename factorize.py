

def main():

    print "DGC Toy"
    for k, vs in Delta_g_DGC.items():
        print k, " = ", factorize_recursive(vs)

    print "\n\nBorromean Rings"
    for k, vs in Delta_g_BR.items():
        print k, " = ", factorize(vs)

    print "\n\n3-Tuples Factoring"
    print factorize_recursive(simple_triples)


def factorize(ts):
    sets = [(frozenset([l]), frozenset([r])) for (l, r) in ts]

    last_size = -1
    cur_size = len(sets)

    while cur_size - last_size:

        left_collect = {frozenset(l): set([]) for (l, _) in sets}
        for (l, r) in sets:
            left_collect[frozenset(l)].symmetric_difference_update(r)

        right_collect = {frozenset(r): set([]) for r in left_collect.values()}
        for l, r in left_collect.items():
            right_collect[frozenset(r)].symmetric_difference_update(l)

        sets = [(frozenset(l), frozenset(r)) for r, l in right_collect.items()]

        last_size = cur_size
        cur_size = len(sets)

    return sets


def factorize3(ts):

    sets = [(frozenset([l]), frozenset([(m, r)])) for (l, m, r) in ts]

    last_size = -1
    cur_size = len(sets)

    while cur_size - last_size:

        left_collect = {l: set([]) for (l, _) in sets}
        for (l, r) in sets:
            left_collect[l].symmetric_difference_update(r)
        #print left_collect
        left_collect = {l: factorize(list(r)) for l, r in left_collect.items()}
        #print left_collect
        right_collect = {frozenset(r): set([]) for r in left_collect.values()}
        for l, r in left_collect.items():
            right_collect[frozenset(r)].symmetric_difference_update(l)

        sets = [(frozenset(l), r) for r, l in right_collect.items()]

        last_size = cur_size
        cur_size = len(sets)

    # flatten everything out
    return [(l, m, r) for (l, mr) in sets for (ms, rs) in mr for m in ms for r in rs]


def factorize_recursive(tps):

    if not isinstance(tps[0], tuple) or len(tps[0]) < 2:
        return tps

    sets = []
    if len(tps[0]) == 2:
        sets = [(frozenset([tp[0]]), frozenset([tp[1]])) for tp in tps]
    else:
        sets = [(frozenset([tp[0]]), frozenset([tp[1:]])) for tp in tps]

    last_size = -1
    cur_size = len(sets)

    while cur_size - last_size:

        left_collect = {l: set([]) for (l, _) in sets}
        for (l, r) in sets:
            left_collect[l].symmetric_difference_update(r)

        left_collect = {l: factorize_recursive(list(r)) for l, r in left_collect.items()}

        right_collect = {frozenset(r): set([]) for r in left_collect.values()}
        for l, r in left_collect.items():
            right_collect[frozenset(r)].symmetric_difference_update(l)

        sets = [(frozenset(l), r) for r, l in right_collect.items()]

        last_size = cur_size
        cur_size = len(sets)


    # flatten everything out
    if len(tps[0]) == 2:
        return sets
    else:
        return [(l,) + t for (l, r) in sets for t in r]


Delta_g_DGC = {'h1_0': [('v', 'a'), ('a', 'v')], 'h0_0': [('v', 'v')], 'h2_0': [('v', 'ab'), ('a', 'b'), ('ab', 'v')]}


Delta_g_BR = {'h0_0': [('v_{1}', 'v_{1}')],
              'h2_1': [('m_{6}', 'c_{9}'), ('c_{6}', 'm_{8}'), ('v_{6}', 't_{5}'), ('c_{10}', 'm_{8}'),
                       ('v_{6}', 't_{6}'), ('m_{9}', 'c_{6}'), ('c_{9}', 'm_{8}'), ('m_{9}', 'c_{9}'),
                       ('c_{6}', 'm_{7}'), ('v_{6}', 't_{8}'), ('c_{10}', 'm_{7}'), ('t_{8}', 'v_{10}'),
                       ('c_{5}', 'm_{7}'), ('c_{9}', 'm_{7}'), ('m_{6}', 'c_{5}'), ('v_{6}', 't_{7}'),
                       ('m_{6}', 'c_{6}'), ('t_{5}', 'v_{10}'), ('m_{9}', 'c_{5}'), ('m_{9}', 'c_{10}'),
                       ('t_{6}', 'v_{10}'), ('t_{7}', 'v_{10}'), ('m_{6}', 'c_{10}'), ('c_{5}', 'm_{8}')],
              'h2_0': [('c_{8}', 'm_{10}'), ('c_{3}', 'm_{10}'), ('m_{4}', 'c_{4}'), ('m_{4}', 'c_{3}'),
                       ('m_{11}', 'c_{7}'), ('m_{4}', 'c_{8}'), ('v_{1}', 't_{2}'), ('c_{7}', 'm_{5}'),
                       ('v_{1}', 't_{1}'), ('c_{3}', 'm_{5}'), ('c_{4}', 'm_{5}'), ('m_{11}', 'c_{8}'),
                       ('m_{4}', 'c_{7}'), ('c_{7}', 'm_{10}'), ('t_{3}', 'v_{5}'), ('t_{2}', 'v_{5}'),
                       ('t_{4}', 'v_{5}'), ('v_{1}', 't_{4}'), ('m_{11}', 'c_{3}'), ('m_{11}', 'c_{4}'),
                       ('c_{8}', 'm_{5}'), ('c_{4}', 'm_{10}'), ('v_{1}', 't_{3}'), ('t_{1}', 'v_{5}')],
              'h1_0': [('v_{1}', 'm_{11}'), ('m_{11}', 'v_{2}'), ('m_{4}', 'v_{2}'), ('v_{1}', 'm_{4}')],
              'h1_1': [('c_{7}', 'v_{3}'), ('v_{2}', 'c_{3}'), ('c_{3}', 'v_{3}'), ('v_{2}', 'c_{7}')],
              'h1_2': [('m_{9}', 'v_{7}'), ('m_{6}', 'v_{7}'), ('v_{6}', 'm_{9}'), ('v_{6}', 'm_{6}')]
              }

simple_triples = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2),
                  (1, 3, 1), (1, 3, 2), (2, 3, 1), (2, 3, 2)]

main()
