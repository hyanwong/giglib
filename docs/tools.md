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

There are no tools yet. Planned tools include converting simple GIGs with structural
variants into tree sequences
([#139](https://github.com/hyanwong/giglib/issues/139)).

```{eval-rst}
.. automodule:: giglib.tools
```
