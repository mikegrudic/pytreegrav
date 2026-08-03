API Documentation
=================

.. automodule:: pytreegrav.frontend
   :noindex:
   :members:

Direct summation
----------------

``Potential`` and ``Accel`` fall back to exact direct summation for small particle counts, and can be
forced to it with ``method="bruteforce"``. Two parallel implementations exist and the frontend picks
between them automatically:

* below ``SYMMETRIC_NMIN`` particles, the straightforward kernel, which gives each thread one target
  and therefore evaluates all :math:`N^2` pairs;
* at or above it, the symmetrized kernel below, which evaluates each pair once -- half the flops --
  at the cost of per-thread scratch buffers.

You normally do not need to call these directly; they are documented because the crossover and the
memory cost are worth knowing about.

.. automodule:: pytreegrav.bruteforce_symmetric
   :noindex:
   :members:

.. automodule:: pytreegrav.bruteforce
   :noindex:
   :members:
