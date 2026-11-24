"""
Simple simulators that record their genealogies as GIGs, one in each module of this
package, e.g. :mod:`giglib.tools.simulation.no_recombination`. Code shared by several
simulators, such as the :class:`~giglib.tools.simulation.dtwf.DTWF_simulator` base class
for forward-time simulators, is in modules of its own.

This package is experimental: see :ref:`sec_tools`. The simulators show how a simulator can
use the tables of a GIG as it runs, e.g. using :meth:`giglib.Tables.find_mrca_regions` to
find homologous positions in two parental genomes. Their results are used in the tests.
"""
