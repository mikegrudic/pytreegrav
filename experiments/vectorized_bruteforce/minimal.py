"""Minimal numba loops: is it stores, reductions, or the combination?"""

import re
import numpy as np
from numba import njit

PD = re.compile(r"^\s+v?\w*(?:add|sub|mul|div|sqrt|fmadd|max|min)\w*pd\b", re.M)
"""Count packed (vector) floating-point instructions in a function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""
"""Count packed (vector) FP instructions in a compiled function's assembly."""


def pk(f):
    """Count packed (vector) FP instructions in a compiled function's assembly."""
    return len(PD.findall("\n".join(f.inspect_asm().values())))


@njit(fastmath=True)  # S1: pure store
def S1(a, b, out):
    """Pure store loop: vectorizes."""
    for j in range(a.shape[0]):
        out[j] = a[j] * b[j]


@njit(fastmath=True)  # S2: pure reduction
def S2(a, b):
    """Pure reduction loop: vectorizes."""
    s = 0.0
    for j in range(a.shape[0]):
        s += a[j] * b[j]
    return s


@njit(fastmath=True)  # S3: reduction + store, same loop
def S3(a, b, out):
    """Reduction plus a plain store in one loop: vectorizes."""
    s = 0.0
    for j in range(a.shape[0]):
        s += a[j] * b[j]
        out[j] = a[j] - b[j]
    return s


@njit(fastmath=True)  # S4: reduction + read-modify-write store
def S4(a, b, out):
    """Reduction plus a read-modify-write store: vectorizes."""
    s = 0.0
    for j in range(a.shape[0]):
        s += a[j] * b[j]
        out[j] -= a[j] * b[j]
    return s


@njit(fastmath=True)  # S5: 3 reductions + 3 RMW stores (the real shape)
def S5(a, b, c, ox, oy, oz):
    """Three reductions plus three RMW stores -- the symmetric kernel's shape: vectorizes."""
    sx = sy = sz = 0.0
    for j in range(a.shape[0]):
        sx += a[j]
        sy += b[j]
        sz += c[j]
        ox[j] -= a[j]
        oy[j] -= b[j]
        oz[j] -= c[j]
    return sx, sy, sz


n = 10000
a = np.random.rand(n)
b = np.random.rand(n)
c = np.random.rand(n)
o1, o2, o3 = np.zeros(n), np.zeros(n), np.zeros(n)
S1(a, b, o1)
S2(a, b)
S3(a, b, o1)
S4(a, b, o1)
S5(a, b, c, o1, o2, o3)
for nm, f in (
    ("S1 store only", S1),
    ("S2 reduction only", S2),
    ("S3 reduction + store", S3),
    ("S4 reduction + RMW store", S4),
    ("S5 3 reductions + 3 RMW", S5),
):
    p = pk(f)
    print(f"  {nm:26s} packed {p:4d}   {'VECTORIZED' if p > 3 else 'not vectorized'}")
