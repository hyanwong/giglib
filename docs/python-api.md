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

:::{currentmodule} giglib
:::

(sec_python_api)=

# Python API

This page provides formal documentation for the {ref}`giglib<sec_welcome>` Python API.

## Helper classes

```{eval-rst}
.. autoclass:: giglib.Segment
  :members:

.. autoclass:: giglib.TskitNameWarning

.. py:data:: UNKNOWN_TIME
    :module: giglib

    The value used for an unknown mutation time (identical to
    :data:`tskit:tskit.UNKNOWN_TIME`). As this is a NaN, test for it using
    :func:`giglib.is_unknown_time`, not ``==``.

.. autofunction:: giglib.is_unknown_time
```


## The **Tables** class

:::{note}
Some `Tables` methods require the data stored in the tables
(in particular that in the {class}`tables.IEdgeTable`) to
conform to certain validity conditions before running. See
{ref}`sec_python_api_tables_validity`.
:::

```{eval-rst}
.. autoclass:: giglib.Tables
  :members:
```

(sec_python_api_tables_validity)=

### Validity flags for **Tables**

:::{seealso}
Validity requirements for a GIG are described in {meth}`Tables.graph`.
:::

```{eval-rst}
.. autoclass:: ValidFlags
  :members:
```

## Specific tables

### The **IEdgeTable** class

```{eval-rst}
.. autoclass:: giglib.tables.IEdgeTable
  :members:
```

### The **NodeTable** class

```{eval-rst}
.. autoclass:: giglib.tables.NodeTable
  :members:
```

### The **IndividualTable** class

```{eval-rst}
.. autoclass:: giglib.tables.IndividualTable
  :members:
```

### The **MutationTable** class

```{eval-rst}
.. autoclass:: giglib.tables.MutationTable
  :members:
```

### Table rows

```{eval-rst}
.. autoclass:: giglib.tables.IEdgeTableRow
  :members:

.. autoclass:: giglib.tables.NodeTableRow
  :members:

.. autoclass:: giglib.tables.IndividualTableRow
  :members:

.. autoclass:: giglib.tables.MutationTableRow
```

## The **Graph** class

```{eval-rst}
.. autoclass:: giglib.Graph
  :members:
```

### Graph objects

Accessors on a {class}`Graph` return objects rather than IDs. Collections of these
objects are returned as a sequence-like view, which supports ``len()``, indexing,
and iteration.

```{eval-rst}
.. autoclass:: giglib.graph.Node
  :members:

.. autoclass:: giglib.graph.IEdge
  :members:

.. autoclass:: giglib.graph.Individual
  :members:

.. autoclass:: giglib.graph.Mutation
  :members:

.. autoclass:: giglib.graph.GIGItemIterator
```
