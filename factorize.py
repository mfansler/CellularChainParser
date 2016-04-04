

def main():
    for k, vs in Delta_g_BR.items():
        print k, " = ", factorize(vs)


def factorize(ts):
    sets = [(set([l]), set([r])) for (l, r) in ts]

    last_size = -1
    cur_size = len(sets)

    while cur_size - last_size:

        left_collect = {frozenset(l): set([]) for (l, _) in sets}
        for (l, r) in sets:
            left_collect[frozenset(l)].symmetric_difference_update(r)

        right_collect = {frozenset(r): set([]) for r in left_collect.values()}
        for l, r in left_collect.items():
            right_collect[frozenset(r)].symmetric_difference_update(l)

        sets = [(set(l), set(r)) for r, l in right_collect.items()]

        last_size = cur_size
        cur_size = len(sets)

    return sets


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

main()
