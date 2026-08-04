"""Optional CUDA backend: ray-traced column density, and monopole gravity.

Deliberately not imported by ``pytreegrav/__init__.py``, so ``import pytreegrav`` works with no CUDA
installed.  Requires ``pip install pytreegrav[cuda]``.  Either hold a context, which packs and uploads
the tree once and is the point for repeated evaluation::

    from pytreegrav.cuda import CudaColumnDensity, CudaPotential, CudaAccel
    ctx = CudaColumnDensity(tree);  columns = ctx(pos, rays)
    ctx = CudaPotential(tree);      phi = ctx(pos, softening, theta=0.7)

or pass ``device="cuda"`` to ColumnDensity/Potential/Accel, which uploads on every call.

Measured on an RTX A6000 against the shipped grouped CPU walks on 32 Xeon Gold 6244 threads, on a real
STARFORGE snapshot (24.7M gas particles): 8.2x for 6-ray column density, and for monopole gravity 8.4x
(potential) / 6.4x (acceleration) with the context reused, 3.7x / 3.6x single-shot -- gravity walks are
short enough that the one-off tree upload is a large share of one call.  Smooth
synthetic clouds give up to 18x, so quote the former -- clustered data puts very unequal walks in one
warp, and the packed tree swamps a 6 MB L2.  Kernels compile to 35-44 registers and zero local memory
(97-100% occupancy) because the walk is stackless: its whole state is one integer cursor threaded
through NextBranch/FirstSubnode, where a conventional traversal would need a per-thread stack in local
memory.

Common to all three: one independent cursor per thread, so no cooperation between lanes and no warp
intrinsics.  Grouping is left to the warp -- 32 Morton-adjacent targets hold the same node index for
most of the descent, so the row fetch broadcasts.  That is the amortization the CPU's grouped walks buy
by hand, but without the acceptance-radius padding they pay for it (which drives column density's leaf
tests from 24 to 1389 per target).

One thing the ray walk must do differently from the CPU: the impact parameter is ``|d - (d.n)n|^2``,
not ``r^2 - z^2``.  That subtraction cancels for a nearly radial ray with relative error ~eps (r/b)^2 -- a
harmless 3.6e-12 in float64, but measured 2.0e-02 in float32.  The stable form is a prerequisite for
float32, not an optimization.  See the per-quantity sections below for accuracy figures.
"""

import math

import numpy as np
from numba import njit

__all__ = ["CudaAccel", "CudaColumnDensity", "CudaPotential", "is_available", "pack_tree", "pack_tree_gravity"]

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


# --------------------------------------------------------------------------------------------------
# Gravity.  Same stackless per-target walk, monopole only.
#
# float32 is safe here, measured not assumed.  Measured error against the float64 walk on a STARFORGE
# snapshot: potential 5.5e-08 median / 1.2e-03 worst, accel 5.7e-07 / 5.3e-03.  The tail is not
# roundoff -- it is the acceptance test flipping for a node on the opening-angle boundary, so it is
# bounded by theta's own ~2e-03 truncation error rather than by epsilon.  Potential fares better
# because every term shares a sign (no cancellation); accel's is a vector residual.
#
# Row layout, 8 float32 = 32 B:
#   0,1,2  centre of mass    3  mass    4  softening    5  size    6  delta    7  pad
# theta is a runtime argument, not folded into the row, so one packed tree serves any opening angle.
# --------------------------------------------------------------------------------------------------

GRAV_ROW = 8


def pack_tree_gravity(tree, dtype=np.float32):
    """Repack an Octree for the gravity kernels: (nodes (NumNodes, 8) ``dtype``, links int32).

    Monopole only -- quadrupoles would need six more slots and a second cache line per node, and the
    measured float32 headroom means monopole at a smaller theta is usually the better trade.
    """
    n = tree.NumNodes
    nodes = np.zeros((n, GRAV_ROW), dtype=np.float64)
    nodes[:, 0:3] = tree.Coordinates[:n]
    nodes[:, 3] = tree.Masses[:n]
    nodes[:, 4] = tree.Softenings[:n]
    nodes[:, 5] = tree.Sizes[:n]
    nodes[:, 6] = tree.Deltas[:n]

    with np.errstate(over="ignore"):
        packed = nodes.astype(dtype)
    if not np.isfinite(packed).all():
        bad = np.argwhere(~np.isfinite(packed))
        raise ValueError(
            f"packing overflowed {np.dtype(dtype).name} at {len(bad)} entries (first: element "
            f"{bad[0][0]}, slot {bad[0][1]}, value {nodes[bad[0][0], bad[0][1]]:.3e})."
        )

    links = np.empty((n, 2), dtype=np.int32)
    links[:, 0] = tree.NextBranch[:n]
    links[:, 1] = tree.FirstSubnode[:n]
    return packed, links


def _potential_kernel(r, h):
    """-1/r softened by the M4 cubic spline; mirrors kernel.PotentialKernel."""
    if h == 0.0:
        return -1.0 / r
    hinv = 1.0 / h
    q = r * hinv
    if q <= 0.5:
        return (-2.8 + q * q * (5.333333333333333 + q * q * (6.4 * q - 9.6))) * hinv
    elif q <= 1.0:
        return (
            -3.2
            + 0.06666666666666667 / q
            + q * q * (10.666666666666666 + q * (-16.0 + q * (9.6 - 2.1333333333333333 * q)))
        ) * hinv
    return -1.0 / r


def _force_kernel(r, h):
    """Enclosed-mass/r^3 for the M4 cubic spline; mirrors kernel.ForceKernel."""
    if r > h:
        return 1.0 / (r * r * r)
    hinv = 1.0 / h
    q = r * hinv
    h3inv = hinv * hinv * hinv
    if q <= 0.5:
        return (10.666666666666666 + q * q * (-38.4 + 32.0 * q)) * h3inv
    return (
        21.333333333333332
        - 48.0 * q
        + 38.4 * q * q
        - 10.666666666666666 * q * q * q
        - 0.06666666666666667 / (q * q * q)
    ) * h3inv


_pot_kernel_cpu = njit(_potential_kernel, inline="always", fastmath=True)
_force_kernel_cpu = njit(_force_kernel, inline="always", fastmath=True)


def _make_grav_walks(pot_kernel, force_kernel):
    """Build (potential, accel) walk bodies bound to the given softening kernels.

    A factory so the same source can be closed over the CPU-jitted kernels or the CUDA device ones --
    numba will not let a cuda device function call an njit one.
    """

    def walk_potential(px, py, pz, soft_t, nodes, links, npart, inv_theta):
        """Potential at (px,py,pz); mirrors treewalk.PotentialWalk, monopole.

        The self term is dropped by ``r > 0``, so the target must be *bit-identical* to its own stored
        row -- hence callers narrowing both with the same float32 cast.  float64 targets against
        float32 rows put a particle 7.9e-08 from itself, inside its softening, adding a spurious
        ``-2.8 m/h``: 1.1e-02 error against 3.1e-08 when consistent.  Accel hides it (the softened
        kernel scales it by a vanishing dx), so the potential is where it surfaces.
        """
        phi = 0.0
        no = npart
        while no > -1:
            dx = nodes[no, 0] - px
            dy = nodes[no, 1] - py
            dz = nodes[no, 2] - pz
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            h = max(nodes[no, 4], soft_t)
            if no < npart:  # leaf
                if r > 0.0:  # neglect the self-potential
                    if r < h:
                        phi += nodes[no, 3] * pot_kernel(r, h)
                    else:
                        phi -= nodes[no, 3] / r
                no = links[no, 0]
            else:
                size = nodes[no, 5]
                delta = nodes[no, 6]
                if r > max(size * inv_theta + delta, h + size * 0.6 + delta):
                    phi -= nodes[no, 3] / r
                    no = links[no, 0]
                else:
                    no = links[no, 1]
        return phi

    def walk_accel(px, py, pz, soft_t, nodes, links, npart, inv_theta):
        """Acceleration at (px,py,pz); mirrors treewalk.AccelWalk, monopole."""
        ax = 0.0
        ay = 0.0
        az = 0.0
        no = npart
        while no > -1:
            dx = nodes[no, 0] - px
            dy = nodes[no, 1] - py
            dz = nodes[no, 2] - pz
            r2 = dx * dx + dy * dy + dz * dz
            r = math.sqrt(r2)
            h = max(nodes[no, 4], soft_t)
            if no < npart:  # leaf
                if r > 0.0:  # no self-force
                    if r < h:
                        fac = nodes[no, 3] * force_kernel(r, h)
                    else:
                        fac = nodes[no, 3] / (r * r2)
                    ax += fac * dx
                    ay += fac * dy
                    az += fac * dz
                no = links[no, 0]
            else:
                size = nodes[no, 5]
                delta = nodes[no, 6]
                if r > max(size * inv_theta + delta, h + size * 0.6 + delta):
                    fac = nodes[no, 3] / (r * r2)
                    ax += fac * dx
                    ay += fac * dy
                    az += fac * dz
                    no = links[no, 0]
                else:
                    no = links[no, 1]
        return ax, ay, az

    return walk_potential, walk_accel


_walk_pot_py, _walk_acc_py = _make_grav_walks(_pot_kernel_cpu, _force_kernel_cpu)
_walk_pot_cpu = njit(_walk_pot_py, fastmath=True)
_walk_acc_cpu = njit(_walk_acc_py, fastmath=True)


@njit(fastmath=True)
def potential_packed_cpu(pos, soft, nodes, links, npart, inv_theta):
    """CPU driver over the packed gravity tree.  The reference the CUDA path is validated against."""
    out = np.empty(pos.shape[0])
    for i in range(pos.shape[0]):
        out[i] = _walk_pot_cpu(pos[i, 0], pos[i, 1], pos[i, 2], soft[i], nodes, links, npart, inv_theta)
    return out


@njit(fastmath=True)
def accel_packed_cpu(pos, soft, nodes, links, npart, inv_theta):
    """CPU driver over the packed gravity tree, acceleration."""
    out = np.empty((pos.shape[0], 3))
    for i in range(pos.shape[0]):
        ax, ay, az = _walk_acc_cpu(pos[i, 0], pos[i, 1], pos[i, 2], soft[i], nodes, links, npart, inv_theta)
        out[i, 0] = ax
        out[i, 1] = ay
        out[i, 2] = az
    return out


_GRAV_KERNELS = None


def _grav_kernels():
    """Compile the two CUDA gravity kernels, lazily."""
    global _GRAV_KERNELS
    if _GRAV_KERNELS is None:
        from numba import cuda

        pk = cuda.jit(_potential_kernel, device=True, inline=True)
        fk = cuda.jit(_force_kernel, device=True, inline=True)
        wp, wa = _make_grav_walks(pk, fk)
        wp = cuda.jit(wp, device=True, inline=True)
        wa = cuda.jit(wa, device=True, inline=True)

        @cuda.jit(fastmath=True)
        def kpot(pos, soft, nodes, links, npart, inv_theta, out):
            i = cuda.grid(1)
            if i < pos.shape[0]:
                out[i] = wp(pos[i, 0], pos[i, 1], pos[i, 2], soft[i], nodes, links, npart, inv_theta)

        @cuda.jit(fastmath=True)
        def kacc(pos, soft, nodes, links, npart, inv_theta, out):
            i = cuda.grid(1)
            if i < pos.shape[0]:
                ax, ay, az = wa(pos[i, 0], pos[i, 1], pos[i, 2], soft[i], nodes, links, npart, inv_theta)
                out[i, 0] = ax
                out[i, 1] = ay
                out[i, 2] = az

        _GRAV_KERNELS = (kpot, kacc)
    return _GRAV_KERNELS


class _CudaGravity:
    """Shared base: packs and uploads a tree once, then evaluates per call."""

    _WIDTH = 1

    def __init__(self, tree, threads_x=THREADS_X):
        from numba import cuda

        if not is_available():
            raise RuntimeError(
                "no CUDA device available. pytreegrav.cuda needs numba-cuda (pip install "
                "pytreegrav[cuda]) and a visible NVIDIA GPU."
            )
        nodes, links = pack_tree_gravity(tree)
        self.npart = int(tree.NumParticles)
        self.threads_x = int(threads_x)
        self.d_nodes = cuda.to_device(nodes)
        self.d_links = cuda.to_device(links)
        self.device_bytes = nodes.nbytes + links.nbytes

    def _run(self, kernel, pos, softening, G, theta, width):
        from numba import cuda

        # same cast the rows got, so a self-gravity target lands exactly on its own leaf and r == 0
        # drops it -- see walk_potential.
        pos = np.ascontiguousarray(pos, dtype=np.float32)
        n = pos.shape[0]
        if softening is None:
            soft = np.zeros(n, dtype=np.float32)
        else:
            soft = np.ascontiguousarray(np.broadcast_to(np.asarray(softening, dtype=np.float32), (n,)))
        shape = (n,) if width == 1 else (n, width)
        d_out = cuda.device_array(shape, dtype=np.float32)
        tx = self.threads_x
        kernel[(n + tx - 1) // tx, tx](
            cuda.to_device(pos),
            cuda.to_device(soft),
            self.d_nodes,
            self.d_links,
            self.npart,
            np.float32(1.0 / theta),
            d_out,
        )
        return G * d_out.copy_to_host()


class CudaPotential(_CudaGravity):
    """Monopole gravitational potential on a CUDA device, against one uploaded tree.

    float32; the measured error against the float64 CPU walk is ~1e-06 relative, well under the
    opening-angle truncation error, and potential has no cancellation at all (every term is negative).
    Hold the instance and call it repeatedly -- packing and uploading the tree is the fixed cost.
    """

    def __call__(self, pos, softening=None, G=1.0, theta=0.7):
        """Potential at ``pos``; ``softening`` is the per-target softening (scalar or array)."""
        return self._run(_grav_kernels()[0], pos, softening, G, theta, 1)


class CudaAccel(_CudaGravity):
    """Monopole gravitational acceleration on a CUDA device, against one uploaded tree.

    float32.  See the note above the gravity section on why that is safe: acceleration is a vector
    residual, but the measured cancellation factor on real data is 2.3 median and 60 worst, leaving
    the float32 error 1-3 orders of magnitude below the theta truncation error.
    """

    def __call__(self, pos, softening=None, G=1.0, theta=0.7):
        """Acceleration at ``pos``, shape (N, 3)."""
        return self._run(_grav_kernels()[1], pos, softening, G, theta, 3)
