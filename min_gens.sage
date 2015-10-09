"""
  Description: A collection of utilities for manipulating CHomP output.
"""

__author__ = "Merv Fansler"


# extracts generators from CHomP output
def extract_generators(hom):
    return {key: val[1] if type(val) is tuple else [0] for key, val in hom.items()}


# iteratively expands a generator using adjacent boundary, then returns the generator with a minimum of cells
def minimize_generator(gen, diff, steps=3):
    gen.set_immutable()
    gens = set([gen])
    for _ in range(steps):
        old_gens = gens.copy()
        expanded = [[c + g for c in diff.columns() if not g.pairwise_product(c).is_zero()] for g in gens]
        for gs in expanded:
            for g in gs:
                g.set_immutable()
                gens.add(g)
        if old_gens == gens:
            break
    return min(sorted(gens, reverse=True), key=lambda c: sum(c.change_ring(ZZ)))


# attempts to reduce all generators in homology
def minimal_generators(gens, diffs, max_steps=3):
    n = max(diffs.keys())
    return {key: gs if gs == [0] else [minimize_generator(g, diffs[key-1], max_steps) for g in gs] for key, gs in gens.items()}