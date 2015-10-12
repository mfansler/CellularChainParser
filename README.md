# CellularChainParser
A set of tools for parsing definitions of graded (co)algebras.

## Introduction
The purpose of this tool set is to provide support for importing and validating definitions for cellular chain complexes and diagonals on these complexes.  A small subset of LaTeX tokens and syntax is used for the input file language.  This is intended to simplify the translation between research and publication workflows.  This tool set is developed as part of a joint research project of Ronald Umble (Millersville University), Barbara Nimershiem (Franklin & Marshall College), and Merv Fansler (Millersville University).

## chain2matrix.py
This script parses a chain complex definition, validates consistency of boundary definitions, and outputs a set out **SageMath** object definitions that can be used to import the definition into **SageMath**.

### Example

    python chain2matrix.py data/mickey.tex

Output will include:

 - differentials (`d1,d2,...`);
 - variable declarations for symbolic ring representation support (`var(...)`); and
 - symbolic chain group definitions (`{ 0: [...], 1: [...], ... }`).

## validateCoproduct.py
This script parses a chain complex definition and checks the validity of the diagonal (coproduct) definition.

### Example

    python validateCoproduct.py data/mickey.tex

Output computes `\Delta \partial` and `(1 \otimes \partial + \partial \otimes 1) \Delta` and compares the results.

## min_gens.sage
The homology computed by CHomP in SageMath is not guaranteed to provide minimal (least hamming weight) generators for the equivalence classes. The methods in this script provide a naive means of attempting to search for minimal generators equivalent to those returned by CHomP.

### Example

    cc_hom = exampleChainComplex.homology(generators=True)
    cc_gens = extract_generators(cc_hom)
    cc_min_gens = minimal_generators(cc_gens, steps=10, pruning=True)
    cc_min_gens
