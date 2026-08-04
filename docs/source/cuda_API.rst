.. _cuda:

GPU-accelerated ray-tracing
===========================

The ray-traced column density path has an optional CUDA backend, benchmarked on an NVIDIA RTX A6000
against 32 Xeon Gold 6244 threads:

.. list-table::
   :header-rows: 1

   * - problem
     - speedup
   * - STARFORGE snapshot, 24.7M gas particles, 6 rays (148M walks)
     - **7.9x**
   * - smooth synthetic clouds, :math:`N = 10^5` / :math:`10^6` / :math:`3 \times 10^6`
     - 11.8x / 15.1x / 18.2x

Expect the former on production data. Clustered structure is the harder case for a GPU: a warp can
hold both dense-core and diffuse-gas sightlines at once, so lanes wait on each other, and at 24.7M
particles the packed tree is 1.5 GB against a 6 MB L2.

Accuracy
--------

The kernel is single precision. On real data the error grows with the number of contributions summed
along a sightline, so it is a distribution rather than a single figure. Measured over 12M sightlines
of that snapshot against the float64 CPU result:

.. list-table::
   :header-rows: 1

   * - median
     - p99
     - p99.9
     - p99.99
     - max
   * - 2.0e-6
     - 4.4e-5
     - 1.7e-4
     - 9.1e-4
     - 2.5e-2

The largest errors fall on the *densest* sightlines: 0.009% of entries exceed :math:`10^{-3}`, and
their median column density is 240x the overall median. Those are precisely the sightlines where
:math:`\tau \gg 1` and the answer is "opaque" whatever the last digits say, so this sits comfortably
below the error of the uniform-sphere density model the estimator is built on.

Note the CPU path agrees with direct summation to :math:`\sim 10^{-15}`, so the two are not
interchangeable if you need reproducible digits. Smooth test problems will understate the spread
above by three orders of magnitude, because they have almost no dynamic range in column density.

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
