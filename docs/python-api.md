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

## The **Graph** class

```{eval-rst}
.. autoclass:: giglib.Graph
  :members:
```
