__author__ = 'mfansler'


class Coalgebra(object):
    def __init__(self, groups, differential, coproduct):

        self.groups = groups
        self.differential = differential
        self.coproduct = coproduct

    def topDimension(self):
        return max([int(g) for g in self.groups])

    def derivative(self, chain):
        # initialize differential to zero
        d_chain = {k: 0 for k in range(self.topDimension())}

        for dim, elems in chain.iteritems():
            for elem in elems:
                d_chain[dim - 1] += self.differential[elem]

        return d_chain

    def incidence_matrix(self, dim, sparse=True):
        if dim not in self.groups or dim == 0:
            return "[0]"
        else:
            matrix = {}
            if not sparse:
                matrix = {(i, j): 0 for i in range(len(self.groups[dim-1])) for j in range(len(self.groups[dim]))}
            for j, face in enumerate(self.groups[dim]):
                if face not in self.differential:
                    continue

                for edge, count in self.differential[face].iteritems():
                    i = self.groups[dim-1].index(edge)
                    matrix[(i, j)] = count
            return matrix
        [x[:] for x in [[foo]*10]*10]

    def __mul__(self, other):
        if type(other) is Coalgebra:
            tensor_groups = {i: [] for i in range(2*self.topDimension()+1)}
            for m in range(self.topDimension() + 1):
                for n in range(self.topDimension() + 1):
                    for l in self.groups[m]:
                        tensor_groups[m+n] += [(l, r) for r in self.groups[n]]

            return tensor_groups
