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

(sec_installation)=

# Installation

Python 3.8 or greater is required. As the current GeneticInheritanceGraphLibrary
API is not stable, you are currently encouraged to install the latest version from
[Github](https://github.com/hyanwong/GeneticInheritanceGraphLibrary), as follows


```bash
python -m pip install git+https://github.com/hyanwong/GeneticInheritanceGraphLibrary
```

After this, you should be able to test it in a Python session using something like the following
(assuming you have also installed [msprime](https://tskit.dev/software/msprime.html))

```{code-cell}
import msprime
import GeneticInheritanceGraphLibrary as gigl  # 😁

ts = msprime.sim_ancestry(
    4, sequence_length=100, recombination_rate=0.01, random_seed=1
)
gig = gigl.Graph.from_tree_sequence(ts)
print(len(gig.nodes), "nodes in this GIG")
```