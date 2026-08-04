"""Optional CUDA backend: ray-traced column density, monopole gravity, and brute-force gravity.

Deliberately not imported by ``pytreegrav/__init__.py``, so ``import pytreegrav`` works with no CUDA installed.  Requires ``pip install pytreegrav[cuda]``.  Either hold a context, which packs and uploads the tree (or the sources) once and is the point for repeated evaluation::

    from pytreegrav.cuda import CudaColumnDensity, CudaPotential, CudaAccel
    ctx = CudaColumnDensity(tree);  columns = ctx(pos, rays)
    ctx = CudaPotential(tree);      phi = ctx(pos, softening, theta=0.7)

or pass ``device="cuda"`` to ColumnDensity/Potential/Accel, which uploads on every call.

On an RTX A6000 against the grouped CPU walks on 32 Xeon Gold 6244 threads, on a 22.3M-particle STARFORGE snapshot: ~12x for 6-ray column density, ~21x for monopole gravity with the context reused (~15x single-shot, a gravity walk being short enough that the pack-and-upload is a third of one call), and 387 Gpair/s brute force against roughly 10.  Clustered data is the harder case -- very unequal walks in one warp, and a 6 MB L2 -- so expect these figures rather than the better ones smooth synthetic clouds give.

All three give each thread one independent stackless cursor, its whole state a single integer threaded through NextBranch/FirstSubnode: no warp intrinsics, and none of the local memory a per-thread traversal stack would need.  Grouping is left to the warp, since 32 Morton-adjacent targets hold the same node index for most of the descent and the row fetch broadcasts -- the amortization the CPU's grouped walks buy by hand, without the acceptance-radius padding they pay for it.

Two things not to undo.  The ray walk's impact parameter must be ``|d - (d.n)n|^2``, not the CPU's ``r^2 - z^2``: that subtraction cancels for a nearly radial ray, a harmless 3.6e-12 in float64 but 2.0e-02 in float32.  And float32 is neither automatic nor visible in the source (see ``_numerics``) -- every kernel here once silently did float64 arithmetic on float32 arrays, costing 39-55x on brute force, so check the PTX rather than trusting the array dtypes::

    ptx = list(kernel.overloads.values())[0].inspect_asm(cc)   # expect zero '.f64'
"""

import math

import numpy as np
from numba import float32, float64, njit, prange

__all__ = [
    "CudaAccel",
    "CudaAccelBruteforce",
    "CudaColumnDensity",
    "CudaPotential",
    "CudaPotentialBruteforce",
    "is_available",
    "pack_tree",
    "pack_tree_gravity",
]

FAC_DENSITY = 3.0 / (4.0 * np.pi)
R_EFF_COEFF = 0.8660254037844386  # sqrt(3)/2, the cube half-diagonal, as in treewalk
THREADS_X = 32  # one warp per block; measured best (486 vs 547 ms at 128, N=1e6 on an A6000)


@njit(inline="always", fastmath=True)
def _rsqrt_cpu(x):
    """CPU stand-in for the device rsqrt; njit so the walk bodies can call it from nopython mode."""
    return 1.0 / math.sqrt(x)


def _numerics(cuda_target):
    """``(cast, sqrt, rsqrt)`` for one compilation target, injected into the shared walk bodies.

    Numba types Python float literals as float64 and ``math.sqrt`` as float64->float64, so a body written naively promotes its whole accumulator chain even when every array is float32.  That is invisible on the CPU and catastrophic on a 1:32-FP64 device: the brute-force kernel measured 7.4 Gpair/s that way against 292 with the casts, and the PTX carried zero f32 divides or sqrts.  So every literal goes through ``cast`` and roots come from libdevice's f32 entry points.

    ``rsqrt`` also replaces the divisions: ``m/r`` and ``m/(r*r2)`` become multiplies by rinv and rinv^3, trading ``div.rn.f32`` (~10 instructions) for one SFU op already needed for ``r``.
    """
    if cuda_target:
        from numba.cuda import libdevice

        return float32, libdevice.sqrtf, libdevice.rsqrtf
    return float64, math.sqrt, _rsqrt_cpu


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


@njit(parallel=True)
def _pack_column_rows(coords, soft, mass, size, delta, npart, rows, nxt, first, links):
    """Fill the column-density rows and links in one parallel pass; return the non-finite count.

    Arithmetic happens in the source dtype and lands in ``rows``, so the result is bit-identical to computing wide and narrowing at the end -- but without the float64 scratch buffer, which was 2.1 GB at N=2.2e7 and, being written one strided column at a time, touched every cache line five times.
    """
    bad = 0
    for i in prange(rows.shape[0]):
        rows[i, 0] = coords[i, 0]
        rows[i, 1] = coords[i, 1]
        rows[i, 2] = coords[i, 2]
        h = soft[i]
        if i < npart:  # leaf
            if h > 0:  # a zero-radius particle has no cross-section; leave its row zeroed
                rows[i, 3] = h * h
                rows[i, 4] = 1.0 / h
                rows[i, 5] = FAC_DENSITY * mass[i] / (h * h)
                rows[i, 6] = 1.0 / (h * h)
        else:
            reach = h + R_EFF_COEFF * size[i] + delta[i]
            rows[i, 3] = reach * reach
        links[i, 0] = nxt[i]
        links[i, 1] = first[i]
        for k in range(7):
            if not np.isfinite(rows[i, k]):
                bad += 1
    return bad


def pack_tree(tree, dtype=np.float32):
    """Repack an Octree into (nodes (NumNodes, 8) ``dtype``, links int32 (NumNodes, 2)).

    Links stay in their own int32 array rather than bit-cast into the float row: float32 represents integers exactly only to 2**24, which N > ~1.1e7 would exceed silently.  ``dtype`` is for validation -- float64 rows isolate any algorithmic difference from the CPU walk, float32 then shows what narrowing costs.
    """
    n = tree.NumNodes
    packed = np.zeros((n, ROW), dtype=dtype)
    links = np.empty((n, 2), dtype=np.int32)
    # The leaf prefactor goes as M/h^2, so an absurdly small radius in awkward units can exceed
    # float32's ~1e38.  Fail loudly rather than uploading inf.
    with np.errstate(over="ignore"):  # reported properly just below
        nbad = _pack_column_rows(
            np.asarray(tree.Coordinates[:n], dtype=np.float64),
            np.asarray(tree.Softenings[:n], dtype=np.float64),
            np.asarray(tree.Masses[:n], dtype=np.float64),
            np.asarray(tree.Sizes[:n], dtype=np.float64),
            np.asarray(tree.Deltas[:n], dtype=np.float64),
            tree.NumParticles,
            packed,
            tree.NextBranch[:n],
            tree.FirstSubnode[:n],
            links,
        )
    if nbad:
        i, slot = np.argwhere(~np.isfinite(packed))[0]
        raise ValueError(
            f"packing overflowed {np.dtype(dtype).name} at {nbad} entries (first: element {i}, slot "
            f"{slot}, from softening {tree.Softenings[i]:.3e} and mass {tree.Masses[i]:.3e}; the leaf "
            "prefactor goes as M/h^2). Rescale masses/radii, or use the float64 CPU path."
        )
    return packed, links


# --------------------------------------------------------------------------------------------------
# The walk.  Written once and jitted twice -- CPU below, CUDA device function in _kernel() -- so the
# GPU-free tests exercise the same body the device runs.
# --------------------------------------------------------------------------------------------------


def _make_column_walk(cuda_target):
    """Build the column-density walk body for one target; see _numerics for why it is a factory."""
    f, sqrt, _ = _numerics(cuda_target)
    ZERO, ONE, TWO = f(0.0), f(1.0), f(2.0)

    def walk(px, py, pz, rx, ry, rz, nodes, links, npart):
        """Column density from (px,py,pz) along unit (rx,ry,rz) over a packed tree.

        Same physics as treewalk.ColumnDensityWalk_singleray: a node opens if its reach sphere meets the forward ray; a leaf contributes the uniform-sphere chord, whole when the target lies outside the sphere and only the forward half when inside.
        """
        col = ZERO
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
                    chord = sqrt(ONE - pp2 * nodes[no, 6])
                    if r2 > s3:  # target outside the sphere: the whole chord, if it lies ahead
                        if z > ZERO:
                            col += nodes[no, 5] * TWO * chord
                    else:  # target inside: forward half-chord only
                        col += nodes[no, 5] * (z * nodes[no, 4] + chord)
                no = links[no, 0]
            else:  # node: reach^2 in slot 3
                if r2 < s3 or (z > ZERO and pp2 < s3):
                    no = links[no, 1]
                else:
                    no = links[no, 0]
        return col

    return walk


_walk = _make_column_walk(False)
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
    """Compile the CUDA kernel, reusing _walk as a device function.  Built lazily so this module imports on a machine with no CUDA."""
    global _KERNEL
    if _KERNEL is None:
        from numba import cuda

        walk_dev = cuda.jit(_make_column_walk(True), device=True, inline=True)

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

    Packing and uploading the tree is the fixed cost -- 179 ms for 60 MB at N=1e6, against 507 ms for a 6-ray pass -- so hold the context and call it repeatedly.  Per-call transfers are ~2% of a call.

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

        ``pos`` in the tree's Morton order (``tree.TreewalkIndices``) gives warp coherence; any other order returns the same answer, more slowly.  ``rays`` are normalized here.
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


@njit(parallel=True)
def _pack_grav_rows(coords, mass, soft, size, delta, rows, nxt, first, links):
    """Fill the gravity rows and links in one parallel pass; return the non-finite count.

    See _pack_column_rows on why this replaced five strided writes into a float64 scratch buffer.
    """
    bad = 0
    for i in prange(rows.shape[0]):
        rows[i, 0] = coords[i, 0]
        rows[i, 1] = coords[i, 1]
        rows[i, 2] = coords[i, 2]
        rows[i, 3] = mass[i]
        rows[i, 4] = soft[i]
        rows[i, 5] = size[i]
        rows[i, 6] = delta[i]
        rows[i, 7] = 0.0
        links[i, 0] = nxt[i]
        links[i, 1] = first[i]
        for k in range(7):
            if not np.isfinite(rows[i, k]):
                bad += 1
    return bad


def pack_tree_gravity(tree, dtype=np.float32):
    """Repack an Octree for the gravity kernels: (nodes (NumNodes, 8) ``dtype``, links int32).

    Monopole only -- quadrupoles would need six more slots and a second cache line per node, and the measured float32 headroom means monopole at a smaller theta is usually the better trade.
    """
    n = tree.NumNodes
    packed = np.empty((n, GRAV_ROW), dtype=dtype)
    links = np.empty((n, 2), dtype=np.int32)
    with np.errstate(over="ignore"):
        nbad = _pack_grav_rows(
            np.asarray(tree.Coordinates[:n], dtype=np.float64),
            np.asarray(tree.Masses[:n], dtype=np.float64),
            np.asarray(tree.Softenings[:n], dtype=np.float64),
            np.asarray(tree.Sizes[:n], dtype=np.float64),
            np.asarray(tree.Deltas[:n], dtype=np.float64),
            packed,
            tree.NextBranch[:n],
            tree.FirstSubnode[:n],
            links,
        )
    if nbad:
        i, slot = np.argwhere(~np.isfinite(packed))[0]
        raise ValueError(
            f"packing overflowed {np.dtype(dtype).name} at {nbad} entries (first: element {i}, slot {slot})."
        )
    return packed, links


def _make_softening_kernels(cuda_target):
    """Build the M4-spline potential/force kernels for one target; see _numerics for the why."""
    f, _, _ = _numerics(cuda_target)

    def potential_kernel(r, h):
        """-1/r softened by the M4 cubic spline; mirrors kernel.PotentialKernel."""
        if h == f(0.0):
            return f(-1.0) / r
        hinv = f(1.0) / h
        q = r * hinv
        if q <= f(0.5):
            return (f(-2.8) + q * q * (f(5.333333333333333) + q * q * (f(6.4) * q - f(9.6)))) * hinv
        elif q <= f(1.0):
            return (
                f(-3.2)
                + f(0.06666666666666667) / q
                + q * q * (f(10.666666666666666) + q * (f(-16.0) + q * (f(9.6) - f(2.1333333333333333) * q)))
            ) * hinv
        return f(-1.0) / r

    def force_kernel(r, h):
        """Enclosed-mass/r^3 for the M4 cubic spline; mirrors kernel.ForceKernel."""
        if r > h:
            return f(1.0) / (r * r * r)
        hinv = f(1.0) / h
        q = r * hinv
        h3inv = hinv * hinv * hinv
        if q <= f(0.5):
            return (f(10.666666666666666) + q * q * (f(-38.4) + f(32.0) * q)) * h3inv
        return (
            f(21.333333333333332)
            - f(48.0) * q
            + f(38.4) * q * q
            - f(10.666666666666666) * q * q * q
            - f(0.06666666666666667) / (q * q * q)
        ) * h3inv

    return potential_kernel, force_kernel


_potential_kernel, _force_kernel = _make_softening_kernels(False)
_pot_kernel_cpu = njit(_potential_kernel, inline="always", fastmath=True)
_force_kernel_cpu = njit(_force_kernel, inline="always", fastmath=True)


def _make_grav_walks(pot_kernel, force_kernel, cuda_target):
    """Build (potential, accel) walk bodies bound to the given softening kernels.

    A factory so the same source can be closed over the CPU-jitted kernels or the CUDA device ones -- numba will not let a cuda device function call an njit one -- and so the literals and roots can be typed per target (see _numerics).
    """
    f, sqrt, rsqrt = _numerics(cuda_target)
    ZERO, C06 = f(0.0), f(0.6)

    def walk_potential(px, py, pz, soft_t, nodes, links, npart, inv_theta):
        """Potential at (px,py,pz); mirrors treewalk.PotentialWalk, monopole.

        The self term is dropped by ``r > 0``, so the target must be *bit-identical* to its own stored row -- hence callers narrowing both with the same float32 cast.  float64 targets against float32 rows put a particle 7.9e-08 from itself, inside its softening, adding a spurious ``-2.8 m/h``: 1.1e-02 error against 3.1e-08 when consistent.  Accel hides it (the softened kernel scales it by a vanishing dx), so the potential is where it surfaces.
        """
        phi = ZERO
        no = npart
        while no > -1:
            dx = nodes[no, 0] - px
            dy = nodes[no, 1] - py
            dz = nodes[no, 2] - pz
            r2 = dx * dx + dy * dy + dz * dz
            r = sqrt(r2)
            h = max(nodes[no, 4], soft_t)
            if no < npart:  # leaf
                if r > ZERO:  # neglect the self-potential
                    if r < h:
                        phi += nodes[no, 3] * pot_kernel(r, h)
                    else:
                        phi -= nodes[no, 3] * rsqrt(r2)
                no = links[no, 0]
            else:
                size = nodes[no, 5]
                delta = nodes[no, 6]
                if r > max(size * inv_theta + delta, h + size * C06 + delta):
                    phi -= nodes[no, 3] * rsqrt(r2)
                    no = links[no, 0]
                else:
                    no = links[no, 1]
        return phi

    def walk_accel(px, py, pz, soft_t, nodes, links, npart, inv_theta):
        """Acceleration at (px,py,pz); mirrors treewalk.AccelWalk, monopole."""
        ax = ZERO
        ay = ZERO
        az = ZERO
        no = npart
        while no > -1:
            dx = nodes[no, 0] - px
            dy = nodes[no, 1] - py
            dz = nodes[no, 2] - pz
            r2 = dx * dx + dy * dy + dz * dz
            r = sqrt(r2)
            h = max(nodes[no, 4], soft_t)
            if no < npart:  # leaf
                if r > ZERO:  # no self-force
                    if r < h:
                        fac = nodes[no, 3] * force_kernel(r, h)
                    else:
                        rinv = rsqrt(r2)
                        fac = nodes[no, 3] * rinv * rinv * rinv
                    ax += fac * dx
                    ay += fac * dy
                    az += fac * dz
                no = links[no, 0]
            else:
                size = nodes[no, 5]
                delta = nodes[no, 6]
                if r > max(size * inv_theta + delta, h + size * C06 + delta):
                    rinv = rsqrt(r2)
                    fac = nodes[no, 3] * rinv * rinv * rinv
                    ax += fac * dx
                    ay += fac * dy
                    az += fac * dz
                    no = links[no, 0]
                else:
                    no = links[no, 1]
        return ax, ay, az

    return walk_potential, walk_accel


_walk_pot_py, _walk_acc_py = _make_grav_walks(_pot_kernel_cpu, _force_kernel_cpu, False)
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

        pk_py, fk_py = _make_softening_kernels(True)
        pk = cuda.jit(pk_py, device=True, inline=True)
        fk = cuda.jit(fk_py, device=True, inline=True)
        wp, wa = _make_grav_walks(pk, fk, True)
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

    float32; the measured error against the float64 CPU walk is ~1e-06 relative, well under the opening-angle truncation error, and potential has no cancellation at all (every term is negative). Hold the instance and call it repeatedly -- packing and uploading the tree is the fixed cost.
    """

    def __call__(self, pos, softening=None, G=1.0, theta=0.7):
        """Potential at ``pos``; ``softening`` is the per-target softening (scalar or array)."""
        return self._run(_grav_kernels()[0], pos, softening, G, theta, 1)


class CudaAccel(_CudaGravity):
    """Monopole gravitational acceleration on a CUDA device, against one uploaded tree.

    float32.  See the note above the gravity section on why that is safe: acceleration is a vector residual, but the measured cancellation factor on real data is 2.3 median and 60 worst, leaving the float32 error 1-3 orders of magnitude below the theta truncation error.
    """

    def __call__(self, pos, softening=None, G=1.0, theta=0.7):
        """Acceleration at ``pos``, shape (N, 3)."""
        return self._run(_grav_kernels()[1], pos, softening, G, theta, 3)


# --------------------------------------------------------------------------------------------------
# Brute force.  Exact direct summation, no tree.  One thread per target; sources are staged through
# shared memory a tile at a time so each source load is reused TILE times, giving O(TILE) flop/byte
# instead of O(1).  Useful as an exact reference and for the small-N regime where a tree does not pay;
# it is O(N^2), so the shipped CPU tree beats it above N~1e5 however fast the kernel is.
# --------------------------------------------------------------------------------------------------

TILE = 128  # threads per block and sources per shared-memory tile


_BF_KERNELS = None


def _bf_kernels():
    """Compile the two CUDA brute-force kernels, lazily.

    Written against cuda.* directly rather than through a shared factory: unlike the tree walks there is no CPU counterpart to keep in step (bruteforce.py already has one), so injecting the intrinsics would buy nothing.
    """
    global _BF_KERNELS
    if _BF_KERNELS is not None:
        return _BF_KERNELS
    from numba import cuda

    _, sqrt, rsqrt = _numerics(True)
    pk_py, fk_py = _make_softening_kernels(True)
    pot_kernel = cuda.jit(pk_py, device=True, inline=True)
    force_kernel = cuda.jit(fk_py, device=True, inline=True)
    ZERO = float32(0.0)

    @cuda.jit(fastmath=True)
    def kpot(pos_t, soft_t, x, m, h, out):
        sx = cuda.shared.array(TILE, float32)
        sy = cuda.shared.array(TILE, float32)
        sz = cuda.shared.array(TILE, float32)
        sm = cuda.shared.array(TILE, float32)
        sh = cuda.shared.array(TILE, float32)
        i = cuda.grid(1)
        tid = cuda.threadIdx.x
        n, ns = pos_t.shape[0], x.shape[0]
        px = pos_t[i, 0] if i < n else ZERO
        py = pos_t[i, 1] if i < n else ZERO
        pz = pos_t[i, 2] if i < n else ZERO
        ht = soft_t[i] if i < n else ZERO
        phi = ZERO
        for base in range(0, ns, TILE):
            j = base + tid
            if j < ns:
                sx[tid] = x[j, 0]
                sy[tid] = x[j, 1]
                sz[tid] = x[j, 2]
                sm[tid] = m[j]
                sh[tid] = h[j]
            cuda.syncthreads()
            if i < n:
                for k in range(min(TILE, ns - base)):
                    dx = sx[k] - px
                    dy = sy[k] - py
                    dz = sz[k] - pz
                    r2 = dx * dx + dy * dy + dz * dz
                    r = sqrt(r2)
                    if r > ZERO:  # self term excluded, as in the CPU kernels
                        hij = max(sh[k], ht)
                        if r < hij:
                            phi += sm[k] * pot_kernel(r, hij)
                        else:
                            phi -= sm[k] * rsqrt(r2)
            cuda.syncthreads()
        if i < n:
            out[i] = phi

    @cuda.jit(fastmath=True)
    def kacc(pos_t, soft_t, x, m, h, out):
        sx = cuda.shared.array(TILE, float32)
        sy = cuda.shared.array(TILE, float32)
        sz = cuda.shared.array(TILE, float32)
        sm = cuda.shared.array(TILE, float32)
        sh = cuda.shared.array(TILE, float32)
        i = cuda.grid(1)
        tid = cuda.threadIdx.x
        n, ns = pos_t.shape[0], x.shape[0]
        px = pos_t[i, 0] if i < n else ZERO
        py = pos_t[i, 1] if i < n else ZERO
        pz = pos_t[i, 2] if i < n else ZERO
        ht = soft_t[i] if i < n else ZERO
        ax = ZERO
        ay = ZERO
        az = ZERO
        for base in range(0, ns, TILE):
            j = base + tid
            if j < ns:
                sx[tid] = x[j, 0]
                sy[tid] = x[j, 1]
                sz[tid] = x[j, 2]
                sm[tid] = m[j]
                sh[tid] = h[j]
            cuda.syncthreads()
            if i < n:
                for k in range(min(TILE, ns - base)):
                    dx = sx[k] - px
                    dy = sy[k] - py
                    dz = sz[k] - pz
                    r2 = dx * dx + dy * dy + dz * dz
                    r = sqrt(r2)
                    if r > ZERO:
                        hij = max(sh[k], ht)
                        if r < hij:
                            fac = sm[k] * force_kernel(r, hij)
                        else:
                            rinv = rsqrt(r2)
                            fac = sm[k] * rinv * rinv * rinv
                        ax += fac * dx
                        ay += fac * dy
                        az += fac * dz
            cuda.syncthreads()
        if i < n:
            out[i, 0] = ax
            out[i, 1] = ay
            out[i, 2] = az

    _BF_KERNELS = (kpot, kacc)
    return _BF_KERNELS


class _CudaBruteforce:
    """Shared base for the exact direct-summation contexts: uploads the source particles once."""

    def __init__(self, pos, m, softening=None):
        from numba import cuda

        if not is_available():
            raise RuntimeError(
                "no CUDA device available. pytreegrav.cuda needs numba-cuda (pip install "
                "pytreegrav[cuda]) and a visible NVIDIA GPU."
            )
        pos = np.ascontiguousarray(pos, dtype=np.float32)
        n = pos.shape[0]
        soft = (
            np.zeros(n, np.float32)
            if softening is None
            else np.ascontiguousarray(np.broadcast_to(np.asarray(softening, np.float32), (n,)))
        )
        self.d_x = cuda.to_device(pos)
        self.d_m = cuda.to_device(np.ascontiguousarray(m, dtype=np.float32))
        self.d_h = cuda.to_device(soft)
        self.n_src = n
        self.device_bytes = pos.nbytes + self.d_m.nbytes + soft.nbytes

    def _run(self, kernel, pos, softening, G, width):
        from numba import cuda

        pos = np.ascontiguousarray(pos, dtype=np.float32)
        n = pos.shape[0]
        soft = (
            np.zeros(n, np.float32)
            if softening is None
            else np.ascontiguousarray(np.broadcast_to(np.asarray(softening, np.float32), (n,)))
        )
        d_out = cuda.device_array((n,) if width == 1 else (n, width), dtype=np.float32)
        kernel[(n + TILE - 1) // TILE, TILE](
            cuda.to_device(pos), cuda.to_device(soft), self.d_x, self.d_m, self.d_h, d_out
        )
        return G * d_out.copy_to_host()


class CudaPotentialBruteforce(_CudaBruteforce):
    """Exact direct-summation potential on a CUDA device, against one uploaded source set.

    Targets default to the sources, giving the self-potential with the self term excluded.
    """

    def __call__(self, pos=None, softening=None, G=1.0):
        if pos is None:
            return self._run(_bf_kernels()[0], self.d_x.copy_to_host(), self.d_h.copy_to_host(), G, 1)
        return self._run(_bf_kernels()[0], pos, softening, G, 1)


class CudaAccelBruteforce(_CudaBruteforce):
    """Exact direct-summation acceleration on a CUDA device.  See CudaPotentialBruteforce."""

    def __call__(self, pos=None, softening=None, G=1.0):
        if pos is None:
            return self._run(_bf_kernels()[1], self.d_x.copy_to_host(), self.d_h.copy_to_host(), G, 3)
        return self._run(_bf_kernels()[1], pos, softening, G, 3)
