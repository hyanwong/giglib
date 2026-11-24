(sec_tools_alignment)=

# Alignments

The `giglib.tools.alignment` module converts between GIGs with insertions and deletions,
and tree sequences whose coordinates are those of a multiple sequence alignment
([#139](https://github.com/hyanwong/giglib/issues/139)), with gaps written as `-`. It works
in both directions: see the module description below for how, and what round trips lose.

```python
import giglib.tools.alignment

gig = giglib.tools.alignment.from_tree_sequence(ts_with_multi_char_indels)

# Each sample's haplotype in the GIG is its alignment with the gaps removed
rows = giglib.tools.alignment.alignments(ts_with_multi_char_indels)
for u, row in zip(ts_with_multi_char_indels.samples(), rows):
    sequence = row.replace("-", "")
    assert gig.haplotype(u, right=len(sequence)) == sequence

# ...and converting back gives a tree sequence with the same haplotypes
ts = giglib.tools.alignment.to_tree_sequence(gig)
rows = giglib.tools.alignment.alignments(ts)
for u, row in zip(ts.samples(), rows):
    assert gig.haplotype(u) == row.replace("-", "")
```

```{eval-rst}
.. automodule:: giglib.tools.alignment
  :members:
```
