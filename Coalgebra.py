from itertools import product

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

    def __mul__(self, other):
        if type(other) is Coalgebra:
            tensor_groups = {i: [] for i in range(2*self.topDimension()+1)}

            for m in range(self.topDimension() + 1):
                for n in range(self.topDimension() + 1):
                    for l in self.groups[m]:
                        tensor_groups[m+n] += [(l, r) for r in self.groups[n]]

            return tensor_groups
        else:
            raise TypeError("Argument must be coalgebra")

    def homology(self):

        return self

    def hom(self, other, degree=0):

        return HomModule.hom(self, other, degree)


class Chain:

    def __init__(self, elements, coalgebra):
        self.elements = elements
        self.coalgebra = coalgebra

    def __add__(self, other):
        if type(other) is Chain and other.coalgebra == self.coalgebra:
            return Chain(self.elements + other.elements, self.coalgebra)
        else:
            return None

    def __mul__(self, other):
        if type(other) is Chain:
            product_space = self.coalgebra(other.coalgebra)
            product_elements = list(product(self.elements, other.elements))
            return Chain(product_elements, product_space)
        else:
            return None

    def __str__(self):
        if not self.elements:
            return "0"
        else:
            def format_tuple(t):
                if type(t) is tuple:
                    return u" \u2297 ".join(list(t))
                else:
                    return str(t)

            return u" + ".join([format_tuple(el) for el in self.elements])


class HomModule:

    def __init__(self, a, b, homomorphisms):
        """
        :param a: domain (Algebra) mapping from
        :param b: range (Algebra) mapping to
        :param homomorphisms: set of maps
        :return: HomModule
        """
        self.a = a
        self.b = b
        self.homomorphisms = homomorphisms

    @classmethod
    def hom(cls, a, b, degree):
        homomorphisms = {}
        for n, group in a.groups.iteritems():
            for el in group:
                homomorphisms[el] = b.groups.get(n + degree)
        return cls(a, b, homomorphisms)

    def __str__(self):
        s = "Module of Homomorphisms\n"
        s += "\n".join(["{} -> {}".format(a, b) for a, l in self.homomorphisms.iteritems() for b in l])
        return s

