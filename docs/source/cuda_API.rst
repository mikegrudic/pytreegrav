.. _cuda:

GPU-accelerated ray-tracing
===========================

The ray-traced column density path has an optional CUDA backend. It is measured at **11.8x / 15.1x /
18.2x** over the parallel CPU walk at :math:`N = 10^5 / 10^6 / 3 \times 10^6`, on an NVIDIA RTX A6000
against 32 Xeon Gold 6244 threads.

It is single precision, so expect a relative error around :math:`10^{-5}` rather than the CPU path's
:math:`10^{-15}`. That is far below the error of the uniform-sphere density model the estimator is
built on, but the two are not interchangeable if you need reproducible last digits.

Installation
------------

.. code-block:: bash

    pip install pytreegrav[cuda]

This adds `numba-cuda <https://github.com/NVIDIA/numba-cuda>`_, which supplies :code:`numba.cuda` from
numba 0.62 onwards. Nothing changes for CPU-only users: :code:`pytreegrav.cuda` is never imported by
the package itself, so :code:`import pytreegrav` never requires a GPU.

Single-shot use
---------------

Pass :code:`device="cuda"` to :func:`~pytreegrav.frontend.ColumnDensity`:

.. code-block:: python

    from pytreegrav import ColumnDensity

    columns = ColumnDensity(x, m, h, rays=6, device="cuda")

This repacks and uploads the tree on every call. The upload is the fixed cost — roughly 180 ms for a
:math:`10^6`-particle tree, against roughly 500 ms for a 6-ray pass — so a single call still wins, but
it leaves something on the table.

Only the ray-traced path is supported: pass :code:`rays`, and leave :code:`randomize_rays` off.
Grouping targets requires them to share a ray grid, which per-target randomization breaks. The 6-bin
angular estimator (:code:`rays=None`) has no GPU path.

Repeated evaluation
-------------------

For many ray grids, target subsets, or timesteps against one mass distribution, hold a context. Per
call, transfers are then about 2% of runtime:

.. code-block:: python

    from pytreegrav import ConstructTree
    from pytreegrav.cuda import CudaColumnDensity

    tree = ConstructTree(x, m, h)
    ctx = CudaColumnDensity(tree)               # pack + upload once
    pos = x[tree.TreewalkIndices]               # Morton order gives warp coherence
    columns = ctx(pos, rays)                    # cheap per call

Supplying targets in the tree's Morton order matters for speed but not for correctness: 32 adjacent
targets tracing one direction hold the same node index for most of the descent, so the tree row fetch
broadcasts across the warp. Any other order returns the same answer more slowly.

Checking for a device
---------------------

.. code-block:: python

    from pytreegrav.cuda import is_available

    if is_available():
        ...

API
---

.. automodule:: pytreegrav.cuda
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:
