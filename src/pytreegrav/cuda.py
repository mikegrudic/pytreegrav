"""Optional CUDA backend for ray-traced column density.

Deliberately not imported by ``pytreegrav/__init__.py``, so ``import pytreegrav`` works with no CUDA
installed.  Requires ``pip install pytreegrav[cuda]``.  Either evaluate repeatedly against one
uploaded tree::

    from pytreegrav.cuda import CudaColumnDensity
    ctx = CudaColumnDensity(tree)   # repack + upload once
    columns = ctx(pos, rays)        # cheap per call

or take the single-shot path, ``ColumnDensity(..., device="cuda")``, which pays the upload each time.

Measured on an RTX A6000 against the shipped grouped CPU walk on 32 Xeon Gold 6244 threads: **7.9x**
on a real STARFORGE snapshot (24.7M gas particles, 6 rays), and 11.8x / 15.1x / 18.2x on smooth
synthetic clouds at N = 1e5 / 1e6 / 3e6.  Expect the former; clustered data puts dense-core and
diffuse-gas sightlines in one warp, so lanes wait on each other, and the 1.5 GB packed tree swamps a
6 MB L2.  float32 error on that snapshot: 2e-06 median, 9e-04 at p99.99, 2.5e-02 worst, concentrated
on the densest sightlines -- it grows with the number of contributions summed, so smooth test problems
understate it by ~3 orders of magnitude.

The kernel compiles to 38 registers and zero local memory -- 100% occupancy -- because the walk is
stackless: its entire state is one integer cursor threaded through NextBranch/FirstSubnode.

Two things differ deliberately from the CPU walk:

* **No grouping.**  ColumnDensity_grouped amortizes the descent over 16 targets by padding the
  acceptance reach with the group's extent, which drives leaf tests from 24 to 1389 per target.  A
  warp gives the same amortization 32-fold for free -- 32 Morton-adjacent targets on one ray hold the
  same node index for nearly the whole descent, so the row fetch broadcasts -- and with no padding,
  so every lane tests its own exact ray.  One cursor per thread also means no warp intrinsics.

* **The impact parameter is |d - (d.n)n|^2, not r^2 - z^2.**  That subtraction cancels for a nearly
  radial ray, with relative error ~eps (r/b)^2: a harmless 3.6e-12 in float64, but measured 2.0e-02
  in float32 at N=2e4.  The stable form is a prerequisite for float32, not an optimization; its nine
  extra flops are free on a latency-bound kernel.
"""

import math

import numpy as np
from numba import njit

__all__ = ["CudaColumnDensity", "is_available", "pack_tree"]

FAC_DENSITY = 3.0 / (4.0 * np.pi)
R_EFF_COEFF = 0.8660254037844386  # sqrt(3)/2, the cube half-diagonal, as in treewalk
THREADS_X = 32  # one warp per block; measured best (486 vs 547 ms at 128, N=1e6 on an A6000)

# Packed row.  Leaves and nodes overload slots 3-6, since an element is only ever one or the other.
#   0,1,2  centre of mass          3  leaf h^2 / node (h + 0.866 size + delta)^2
#   4      leaf 1/h                5  leaf 3M/(4 pi h^2)
#   6      leaf 1/h^2              7  pad, so a row is 32 B and two share a 64 B line
# Precomputing the reciprocals and the squared reach leaves the kernel with no divisions at all, and
# no sqrt on the node path (r^2 vs reach^2 rather than r vs reach).
ROW = 8


def is_available():
    """True if numba-cuda is installed and a device is visible."""
    try:
        from numba import cuda
    except ImportError:
        return False
    try:
        return bool(cuda.is_available())
    except cuda.CudaSupportError:  # toolkit present, driver broken or absent -> unavailable, not an error
        return False


def pack_tree(tree, dtype=np.float32):
    """Repack an Octree into (nodes (NumNodes, 8) ``dtype``, links int32 (NumNodes, 2)).

    Links stay in their own int32 array rather than bit-cast into the float row: float32 represents
    integers exactly only to 2**24, which N > ~1.1e7 would exceed silently.  ``dtype`` is for
    validation -- float64 rows isolate any algorithmic difference from the CPU walk, float32 then
    shows what narrowing costs.
    """
    n = tree.NumNodes
    npart = tree.NumParticles
    nodes = np.zeros((n, ROW), dtype=np.float64)  # build wide, narrow at the end
    nodes[:, 0:3] = tree.Coordinates[:n]

    h = np.asarray(tree.Softenings[:n], dtype=np.float64)
    is_leaf = np.arange(n) < npart
    ok = is_leaf & (h > 0)  # a zero-radius particle has no cross-section; leave its row zeroed

    nodes[ok, 3] = h[ok] ** 2
    nodes[ok, 4] = 1.0 / h[ok]
    nodes[ok, 5] = FAC_DENSITY * np.asarray(tree.Masses[:n], dtype=np.float64)[ok] / h[ok] ** 2
    nodes[ok, 6] = 1.0 / h[ok] ** 2

    node_sel = ~is_leaf
    reach = h[node_sel] + R_EFF_COEFF * np.asarray(tree.Sizes[:n], dtype=np.float64)[node_sel]
    reach += np.asarray(tree.Deltas[:n], dtype=np.float64)[node_sel]
    nodes[node_sel, 3] = reach**2

    # The leaf prefactor goes as M/h^2, so an absurdly small radius in awkward units can exceed
    # float32's ~1e38.  Fail loudly rather than uploading inf.
    with np.errstate(over="ignore"):  # reported properly just below
        packed = nodes.astype(dtype)
    if not np.isfinite(packed).all():
        bad = np.argwhere(~np.isfinite(packed))
        raise ValueError(
            f"packing overflowed {np.dtype(dtype).name} at {len(bad)} entries (first: element "
            f"{bad[0][0]}, slot {bad[0][1]}, value {nodes[bad[0][0], bad[0][1]]:.3e}). Rescale "
            "masses/radii, or use the float64 CPU path."
        )

    links = np.empty((n, 2), dtype=np.int32)
    links[:, 0] = tree.NextBranch[:n]
    links[:, 1] = tree.FirstSubnode[:n]
    return packed, links


# --------------------------------------------------------------------------------------------------
# The walk.  Written once and jitted twice -- CPU below, CUDA device function in _kernel() -- so the
# GPU-free tests exercise the same body the device runs.
# --------------------------------------------------------------------------------------------------


def _walk(px, py, pz, rx, ry, rz, nodes, links, npart):
    """Column density from (px,py,pz) along unit (rx,ry,rz) over a packed tree.

    Same physics as treewalk.ColumnDensityWalk_singleray: a node opens if its reach sphere meets the
    forward ray; a leaf contributes the uniform-sphere chord, whole when the target lies outside the
    sphere and only the forward half when inside.
    """
    col = 0.0
    no = npart  # root
    while no > -1:
        dx = nodes[no, 0] - px
        dy = nodes[no, 1] - py
        dz = nodes[no, 2] - pz
        r2 = dx * dx + dy * dy + dz * dz
        z = rx * dx + ry * dy + rz * dz
        # |d - z n|^2, not r2 - z*z -- see the module docstring; a sum of squares, so >= 0 by
        # construction, which is also why no negative-pp2 guard is needed.
        bx = dx - z * rx
        by = dy - z * ry
        bz = dz - z * rz
        pp2 = bx * bx + by * by + bz * bz
        s3 = nodes[no, 3]
        if no < npart:  # leaf: h^2 in slot 3
            if pp2 < s3:
                chord = math.sqrt(1.0 - pp2 * nodes[no, 6])
                if r2 > s3:  # target outside the sphere: the whole chord, if it lies ahead
                    if z > 0.0:
                        col += nodes[no, 5] * 2.0 * chord
                else:  # target inside: forward half-chord only
                    col += nodes[no, 5] * (z * nodes[no, 4] + chord)
            no = links[no, 0]
        else:  # node: reach^2 in slot 3
            if r2 < s3 or (z > 0.0 and pp2 < s3):
                no = links[no, 1]
            else:
                no = links[no, 0]
    return col


_walk_cpu = njit(_walk, fastmath=True)


@njit(fastmath=True)
def column_density_packed_cpu(pos, rays, nodes, links, npart):
    """CPU driver over the packed tree.  The reference the CUDA path is validated against."""
    out = np.empty((pos.shape[0], rays.shape[0]))
    for i in range(pos.shape[0]):
        for r in range(rays.shape[0]):
            out[i, r] = _walk_cpu(
                pos[i, 0], pos[i, 1], pos[i, 2], rays[r, 0], rays[r, 1], rays[r, 2], nodes, links, npart
            )
    return out


_KERNEL = None


def _kernel():
    """Compile the CUDA kernel, reusing _walk as a device function.  Built lazily so this module
    imports on a machine with no CUDA."""
    global _KERNEL
    if _KERNEL is None:
        from numba import cuda

        walk_dev = cuda.jit(_walk, device=True, inline=True)

        @cuda.jit(fastmath=True)
        def kernel(pos, rays, nodes, links, npart, out):
            # x over consecutive (Morton-ordered) targets, y over rays, so a warp is 32 spatially
            # adjacent targets sharing one direction -- they hold the same node index for most of the
            # descent, making the row fetch a broadcast.
            i, r = cuda.grid(2)
            if i < pos.shape[0] and r < rays.shape[0]:
                out[i, r] = walk_dev(
                    pos[i, 0], pos[i, 1], pos[i, 2], rays[r, 0], rays[r, 1], rays[r, 2], nodes, links, npart
                )

        _KERNEL = kernel
    return _KERNEL


class CudaColumnDensity:
    """Ray-traced column density on a CUDA device, against one uploaded tree.

    Packing and uploading the tree is the fixed cost -- 179 ms for 60 MB at N=1e6, against 507 ms for
    a 6-ray pass -- so hold the context and call it repeatedly.  Per-call transfers are ~2% of a call.

    Arguments:
    tree -- Octree holding the source mass distribution
    threads_x -- threads per block along the target axis (default 32, one warp; measured best)
    """

    def __init__(self, tree, threads_x=THREADS_X):
        from numba import cuda

        if not is_available():
            raise RuntimeError(
                "no CUDA device available. pytreegrav.cuda needs numba-cuda (pip install "
                "pytreegrav[cuda]) and a visible NVIDIA GPU."
            )
        nodes, links = pack_tree(tree)
        self.npart = int(tree.NumParticles)
        self.threads_x = int(threads_x)
        self.d_nodes = cuda.to_device(nodes)
        self.d_links = cuda.to_device(links)
        self.device_bytes = nodes.nbytes + links.nbytes

    def __call__(self, pos, rays):
        """Column densities, shape (len(pos), len(rays)), float32.

        ``pos`` in the tree's Morton order (``tree.TreewalkIndices``) gives warp coherence; any other
        order returns the same answer, more slowly.  ``rays`` are normalized here.
        """
        from numba import cuda

        pos = np.ascontiguousarray(pos, dtype=np.float32)
        rays = np.atleast_2d(np.asarray(rays, dtype=np.float64))
        rays = np.ascontiguousarray(rays / np.linalg.norm(rays, axis=1)[:, None], dtype=np.float32)
        n, nr = pos.shape[0], rays.shape[0]

        d_out = cuda.device_array((n, nr), dtype=np.float32)
        tx = self.threads_x
        blocks = ((n + tx - 1) // tx, nr)  # one block row per ray
        _kernel()[blocks, (tx, 1)](
            cuda.to_device(pos), cuda.to_device(rays), self.d_nodes, self.d_links, self.npart, d_out
        )
        return d_out.copy_to_host()
