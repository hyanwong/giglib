(sec_tools_indels)=

# Tree sequences with indels

The `giglib.tools.indels` module converts between GIGs with insertions and deletions
(indels), and tree sequences that represent indels through the coordinates of a multiple
sequence alignment ([#139](https://github.com/hyanwong/giglib/issues/139)), with gaps
written as `-`. It works in both directions: see the module description below for how, and
what round trips lose.

As a multiple sequence alignment gives all the genomes the same coordinates, in the same
order, it can only represent substitutions, insertions and deletions. Other structural
variants, such as inversions, duplications and moved copies (e.g. of transposable elements),
can't be represented in a tree sequence this way, so need a GIG built in other ways, e.g. by
a {ref}`forward-time simulator <sec_tools_simulation>`.

```python
import giglib.tools.indels

gig = giglib.tools.indels.from_tree_sequence(ts_with_multi_char_indels)

# Each sample's haplotype in the GIG is its alignment with the gaps removed
rows = giglib.tools.indels.alignments(ts_with_multi_char_indels)
for u, row in zip(ts_with_multi_char_indels.samples(), rows):
    sequence = row.replace("-", "")
    assert gig.haplotype(u, right=len(sequence)) == sequence

# ...and converting back gives a tree sequence with the same haplotypes
ts = giglib.tools.indels.to_tree_sequence(gig)
rows = giglib.tools.indels.alignments(ts)
for u, row in zip(ts.samples(), rows):
    assert gig.haplotype(u) == row.replace("-", "")
```

```{eval-rst}
.. automodule:: giglib.tools.indels
  :members:
```
