(sec_design)=

# Design notes

These notes record design decisions for _giglib_, and the reasoning behind them. They are
intended for developers and expert users, and are a place to refer back to when implementing or
reviewing features.

Each note starts with a **status**:

- **Proposed**: under discussion, and may change substantially
- **Accepted**: agreed, but not (fully) implemented yet. The code may not match the note.
- **Implemented**: the note describes what the code does

Discussion happens in GitHub issues and pull requests, which are linked from each note. When a
decision changes, update the note (and its status) in the same pull request as the code.

**Performance.** Decisions about speed and memory are based on measurements, made with the
scripts in the [`evaluation/`](https://github.com/hyanwong/giglib/tree/main/evaluation) folder of
the repository. Its [README](https://github.com/hyanwong/giglib/blob/main/evaluation/README.md)
explains how to run them and compare runs, and has a table of findings, each linked to the pull
request that acted on it (e.g. the speed-ups to {meth}`~giglib.Tables.find_homology` in
{pr}`230` and {pr}`234`).

```{tableofcontents}
```
