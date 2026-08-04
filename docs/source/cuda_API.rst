.. _cuda:

GPU acceleration
================

Column density, monopole tree gravity and brute-force gravity have an optional CUDA backend. Measured on
an NVIDIA RTX A6000 against 32 Xeon Gold 6244 threads, on a STARFORGE snapshot with 22.3M gas particles
(33.5M nodes):

.. list-table::
   :header-rows: 1

   * - problem
     - CPU 32t
     - CUDA
     - context reused
     - single call
   * - ``ColumnDensity``, 6 rays (134M walks)
     - 638 s
     - 52 s
     - **12.3x**
     - 12.2x
   * - monopole ``Potential``
     - 19.0 s
     - 0.91 s
     - **20.9x**
     - 14.2x
   * - monopole ``Accel``
     - 21.1 s
     - 0.99 s
     - **21.4x**
     - 14.9x
   * - brute force, pair rate at :math:`N = 2.6 \times 10^5`
     - ~10 Gpair/s
     - 387 Gpair/s
     - **~40x**
     - --

The two columns differ by the one-off pack-and-upload of the tree, 0.43 s for gravity and 0.54 s here:
0.12 s to build the packed rows and the rest PCIe. It is negligible against a 52 s column-density pass and
a third of a 0.9 s gravity walk, which is why gravity in particular still wants a held context. (The
*first* large device allocation in a process costs an extra ~0.6 s on top, once.)

Clustered structure is the harder case for a GPU: a warp can hold both dense-core and diffuse-gas
sightlines at once, so lanes wait on each other, and the packed tree is over 1 GB against a 6 MB L2.
Smooth synthetic clouds do better, so the figures above are the ones to expect on production data.

Accuracy
--------

The kernels are single precision. On real data the error grows with the number of contributions summed
along a sightline, so it is a distribution rather than a single figure. Measured over 1.2M sightlines of
that snapshot, against the same walk in float64:

.. list-table::
   :header-rows: 1

   * - median
     - p99
     - p99.9
     - p99.99
     - max
   * - 1.5e-6
     - 3.7e-5
     - 9.7e-5
     - 2.2e-4
     - 2.3e-2

The largest errors fall on the *densest* sightlines: 0.0003% of entries exceed :math:`10^{-3}`, and their
median column density is ~2000x the overall median. Those are precisely the sightlines where
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

This repacks and uploads the tree on every call, which for column density is a rounding error: 0.54 s
against a 52 s pass at :math:`N = 2.2 \times 10^7`. Holding a context is worth much more for gravity,
whose walks are short enough for that fixed cost to matter.

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

Gravity
-------

``Potential`` and ``Accel`` take the same ``device="cuda"`` flag. Monopole only -- quadrupoles would
need six more slots per node and a second cache line, and the float32 headroom means monopole at a
smaller ``theta`` is usually the better trade:

.. code-block:: python

    from pytreegrav import Potential, Accel

    phi = Potential(x, m, h, theta=0.7, device="cuda")
    acc = Accel(x, m, h, theta=0.7, device="cuda")

Or hold a context, which is the point for repeated evaluation against one mass distribution -- an
N-body step, or a binding-energy sweep over many candidate groups:

.. code-block:: python

    from pytreegrav.cuda import CudaPotential, CudaAccel

    ctx = CudaPotential(tree)          # pack + upload once
    phi = ctx(pos, softening, G=1.0, theta=0.7)

``theta`` is a per-call argument rather than baked into the packed tree, so one upload serves any
opening angle.

Hold the context if you can. Gravity walks are short -- under 0.1 us per particle on the device -- so the
one-off pack-and-upload is a third of a single call: on the snapshot above, 20.9x with the context reused
against 14.2x for a bare ``Potential(..., device="cuda")``.

Accuracy on that snapshot, against two different CPU references, because the choice matters more than
the precision does:

.. list-table::
   :header-rows: 1

   * - quantity
     - CPU reference
     - median
     - p99
     - max
   * - potential
     - same walk, ungrouped
     - 1.3e-7
     - 8.3e-7
     - 7.5e-6
   * - acceleration
     - same walk, ungrouped
     - 2.0e-9
     - 8.6e-7
     - 1.9e-4
   * - potential
     - grouped CPU default
     - 1.8e-5
     - 7.5e-4
     - 2.7e-3
   * - acceleration
     - grouped CPU default
     - 4.9e-6
     - 2.6e-4
     - 5.4e-3

The first two rows are float32 and nothing else: they compare against the *ungrouped* walk, which is the
algorithm the device actually runs. The last two are what you see if you diff ``device="cuda"`` against
the default CPU path, and they are two orders of magnitude larger -- because the CPU groups targets and so
opens a superset of nodes, a difference in the approximation rather than in the arithmetic. All four rows
are unchanged if the device kernels are compiled in float64, which is what establishes that.

The residual tail in the first two rows is the acceptance test flipping for a node sitting on the
opening-angle boundary, which changes the answer by that node's own truncation error -- so it is bounded
by :math:`\theta`'s error (~2e-3 RMS at 0.7), not by machine precision. A flip effectively gives you the
answer for a marginally different :math:`\theta`. If you need an absolute error, compare against brute
force rather than against either CPU tree path.

Brute force
-----------

``method="bruteforce"`` also takes ``device="cuda"``. This is the one place the GPU is at its best:
:math:`O(N^2)` with no traversal, no divergence, and every source shared across a block, so it runs at
the device's arithmetic peak rather than its memory latency.

.. code-block:: python

    phi = Potential(x, m, h, method="bruteforce", device="cuda")

    from pytreegrav.cuda import CudaPotentialBruteforce   # or CudaAccelBruteforce
    ctx = CudaPotentialBruteforce(x_source, m_source, h_source)
    phi = ctx(x_target, h_target, G=1.0)                  # sources stay resident

Measured on an A6000: **387 Gpair/s** for the potential and 331 for the acceleration at
:math:`N = 2.6 \times 10^5`, against roughly 10 Gpair/s on 32 CPU threads -- about **40x**. Sources are
staged through shared memory in tiles of 128, and the self term is dropped by the same ``r > 0`` test the
tree walks use, so targets must be narrowed to float32 consistently with the sources.

The practical consequence is where the crossover moves. Brute force is exact, so it is worth using
whenever it is affordable, and on the GPU it stays cheaper than the *tree* out to
:math:`N \approx 10^5`, against :math:`N \approx 7 \times 10^3` on 32 CPU threads:

.. list-table::
   :header-rows: 1

   * - N
     - CPU 32t tree
     - CPU 32t brute force
     - CUDA tree
     - CUDA brute force
   * - :math:`10^3`
     - 0.56
     - 0.32
     - 1.47
     - 1.07
   * - :math:`2 \times 10^4`
     - 0.81
     - 2.09
     - 0.32
     - **0.18**
   * - :math:`10^5`
     - 0.77
     - --
     - 0.29
     - **0.28**
   * - :math:`2 \times 10^6`
     - 0.84
     - --
     - **0.37**
     - --

(:math:`\mu s` per particle, Plummer, :math:`\theta = 0.7`, potential; see
``examples/benchmark_scaling.py --cuda``.) Note the small-:math:`N` end: below
:math:`N \approx 5 \times 10^3` the GPU is *slower* than 32 CPU threads, because a kernel launch plus a
tree upload is not worth amortizing over that little work.

Accuracy is float32 accumulation over N terms and nothing else -- no :math:`\theta`, so no tail. It grows
as :math:`\sqrt{N}`: the worst case measured (acceleration, unsoftened) is 2.4e-6 at :math:`N = 10^3`,
3.3e-5 at :math:`2 \times 10^4`, and 5.1e-5 at :math:`6 \times 10^4`.

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
