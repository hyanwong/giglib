---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec_concepts)=

# Concepts

Recombinant genealogies in the form of Ancestral Recombination Graphs (ARGs) can't
encode complex genomic rearrangements that generate structural variation (SV).
SV is increasingly seen as important both within and, especially, between species.

A (generalised) genetic inheritance graph (or GIG) extends the concept of an ARG
so that “edge annotations” can specify different parent and child left/right coordinates.
The resulting structure is comparable to an ARG, but is capable of representing
arbitrarily complex patterns of genetic inheritance. 

The GeneticInheritanceGraphLibrary is a proof-of-concept implementation of the idea
behind GIGs, and is heavily based on the standard [tskit](https://tskit.dev) ARG
library.

:::{todo}
Fill out more details from the [README.md](https://github.com/hyanwong/GeneticInheritanceGraphLibrary/blob/main/README.md)
:::
