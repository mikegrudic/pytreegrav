// Does LLVM vectorize the symmetric (scatter-store) direct-summation loop when told the
// pointers do not alias?  accel_sym_plain and accel_sym_restrict are byte-identical apart
// from __restrict.
//
// Result: BOTH vectorize (width 4, interleaved 2) and produce bit-identical output, so
// __restrict is worth nothing here -- LLVM versions the loop with runtime alias checks by
// itself.  This disproves "LLVM cannot prove no-alias" as the explanation for numba's
// scalar codegen, and shows the loop is vectorizable in principle.  Single-threaded, this
// C++ runs the same algorithm ~2.8x faster than numba's best symmetric kernel.
//
// Build and run:
//   clang++ -O3 -march=native -ffast-math -std=c++17 alias_test.cpp -o alias_test
//   ./alias_test 20000
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

// 1/sqrt(x) from multiplies only: magic-constant seed + 4 Newton steps (~7.6e-15 rel).
// Mirrors _rsqrt in the numba experiments so the two are comparable.
static inline double rsqrt4(double x) {
    int64_t i; double y;
    std::memcpy(&i, &x, 8);
    i = 0x5FE6EB50C7B537A9LL - (i >> 1);
    std::memcpy(&y, &i, 8);
    const double xh = 0.5 * x;
    y = y * (1.5 - xh * y * y);
    y = y * (1.5 - xh * y * y);
    y = y * (1.5 - xh * y * y);
    y = y * (1.5 - xh * y * y);
    return y;
}

// ---- A: plain pointers (what numba emits) -------------------------------------------------
__attribute__((noinline))
void accel_sym_plain(const double* px, const double* py, const double* pz, const double* m,
                     double* ox, double* oy, double* oz, int n) {
    for (int i = 0; i < n; ++i) {
        const double xi = px[i], yi = py[i], zi = pz[i], mi = m[i];
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int j = i + 1; j < n; ++j) {          // <-- the loop under test
            const double dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double ri = r2 > 0.0 ? rsqrt4(r2) : 0.0;
            const double k = ri * ri * ri;
            ax += k * m[j] * dx;  ay += k * m[j] * dy;  az += k * m[j] * dz;
            ox[j] -= k * mi * dx; oy[j] -= k * mi * dy; oz[j] -= k * mi * dz;
        }
        ox[i] += ax; oy[i] += ay; oz[i] += az;
    }
}

// ---- B: identical, but the pointers are declared non-aliasing -----------------------------
__attribute__((noinline))
void accel_sym_restrict(const double* __restrict px, const double* __restrict py,
                        const double* __restrict pz, const double* __restrict m,
                        double* __restrict ox, double* __restrict oy,
                        double* __restrict oz, int n) {
    for (int i = 0; i < n; ++i) {
        const double xi = px[i], yi = py[i], zi = pz[i], mi = m[i];
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int j = i + 1; j < n; ++j) {          // <-- the loop under test
            const double dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double ri = r2 > 0.0 ? rsqrt4(r2) : 0.0;
            const double k = ri * ri * ri;
            ax += k * m[j] * dx;  ay += k * m[j] * dy;  az += k * m[j] * dz;
            ox[j] -= k * mi * dx; oy[j] -= k * mi * dy; oz[j] -= k * mi * dz;
        }
        ox[i] += ax; oy[i] += ay; oz[i] += az;
    }
}

// ---- C: non-symmetric gather-only control (vectorizes in numba already) --------------------
__attribute__((noinline))
void accel_nonsym(const double* px, const double* py, const double* pz, const double* m,
                  double* ox, double* oy, double* oz, int n) {
    for (int i = 0; i < n; ++i) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int j = 0; j < n; ++j) {
            const double dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double ri = r2 > 0.0 ? rsqrt4(r2) : 0.0;
            const double k = m[j] * ri * ri * ri;
            ax += k * dx; ay += k * dy; az += k * dz;
        }
        ox[i] = ax; oy[i] = ay; oz[i] = az;
    }
}

// Best-of-`reps` wall time for a nullary callable.
template <class F>
double timeit(F f, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

// Build a random cube of particles, run all three kernels, report timings and check that
// the plain and __restrict symmetric variants agree.
int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 20000;
    std::vector<double> px(n), py(n), pz(n), m(n), ax(n), ay(n), az(n), bx(n), by(n), bz(n);
    srand(1);
    for (int i = 0; i < n; ++i) {
        px[i] = rand() / (double)RAND_MAX; py[i] = rand() / (double)RAND_MAX;
        pz[i] = rand() / (double)RAND_MAX; m[i] = (rand() / (double)RAND_MAX) / n;
    }
    auto zero = [&](std::vector<double>& v) { std::fill(v.begin(), v.end(), 0.0); };

    double t_plain = timeit([&]{ zero(ax); zero(ay); zero(az);
        accel_sym_plain(px.data(),py.data(),pz.data(),m.data(),ax.data(),ay.data(),az.data(),n); }, 3);
    double t_restr = timeit([&]{ zero(bx); zero(by); zero(bz);
        accel_sym_restrict(px.data(),py.data(),pz.data(),m.data(),bx.data(),by.data(),bz.data(),n); }, 3);
    std::vector<double> cx(n), cy(n), cz(n);
    double t_nonsym = timeit([&]{
        accel_nonsym(px.data(),py.data(),pz.data(),m.data(),cx.data(),cy.data(),cz.data(),n); }, 3);

    double maxrel = 0.0, scale = 0.0;
    for (int i = 0; i < n; ++i) {
        scale = std::max(scale, std::fabs(ax[i]));
        maxrel = std::max(maxrel, std::fabs(ax[i] - bx[i]));
    }
    const double pairs_sym = 0.5 * n * (double)(n - 1), pairs_non = (double)n * n;
    printf("N=%d\n", n);
    printf("  symmetric plain    : %8.1f ms   %6.2f Ginteract/s\n", t_plain*1e3, pairs_sym/t_plain/1e9);
    printf("  symmetric restrict : %8.1f ms   %6.2f Ginteract/s   speedup vs plain %.2fx\n",
           t_restr*1e3, pairs_sym/t_restr/1e9, t_plain/t_restr);
    printf("  non-symmetric      : %8.1f ms   %6.2f Ginteract/s   (per-pair equiv %.1f ms)\n",
           t_nonsym*1e3, pairs_non/t_nonsym/1e9, t_nonsym*1e3/2);
    printf("  plain vs restrict agree to %.2e (rel)\n", maxrel/scale);
    return 0;
}
