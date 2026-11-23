(sec_tools)=

# Experimental tools

The `giglib.tools` subpackage contains extra functionality that is built on top of
_giglib_, but is **not part of the core API**:

- **Not imported by default.** `import giglib` does not load it, so you need to import it
  explicitly, e.g. `import giglib.tools`.
- **Experimental.** Tools may change, or be removed, without warning, and may only work
  in restricted cases (raising an error otherwise).
- **Built on the public API.** Tools only use the documented _giglib_ API, so they do not
  depend on how _giglib_ stores its data internally (see {ref}`sec_design_storage`).
- **Tested.** Tools are tested along with the rest of _giglib_.

```{eval-rst}
.. automodule:: giglib.tools
```

(sec_tools_alignment)=

## Alignments

The `giglib.tools.alignment` module converts between GIGs with insertions and deletions,
and tree sequences whose coordinates are those of a multiple sequence alignment
([#139](https://github.com/hyanwong/giglib/issues/139)). We call these the *reference
coordinates*, as they are shared by all the genomes. (They are often called the "columns"
of an alignment, but we avoid that term, so as not to confuse them with the columns of
giglib's tables.) Gaps are written as `-`, and a site can span several reference
coordinates, so that e.g. a deletion of three bases is a single
mutation from `ATCG` to `A---`. Conversion works in both directions, for GIGs whose
rearrangements are insertions and deletions.

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
