# systolic_sim.py
# Cycle-accurate systolic array simulator for matrix multiplication.
#
# Supports two dataflows:
#   1. Output-Stationary (OS): each PE holds a fixed C[i][j] accumulator;
#      A streams right, B streams down.
#   2. Weight-Stationary (WS): each PE holds a fixed weight B[k][n];
#      activations stream right, partial sums stream down and exit at
#      the bottom row as finished C[m][n] values.
#
# Both support rectangular arrays (M x N PEs, not just square N x N),
# with A: (M x K), B: (K x N), C: (M x N).

import random

# ---------------------------------------------------------
# Reference software multiplication (for correctness check)
# ---------------------------------------------------------
def matmul_ref(A, B):
    n = len(A)
    k_dim = len(B)
    m = len(B[0])

    C = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for t in range(k_dim):
                C[i][j] += A[i][t] * B[t][j]
    return C


def random_matrix(rows, cols, lo=-5, hi=5):
    return [[random.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


# ===========================================================
# 1) OUTPUT-STATIONARY DATAFLOW
# ===========================================================
class OS_PE:
    __slots__ = ("a", "b", "psum", "active_cycles", "next_a", "next_b")

    def __init__(self):
        self.a = None
        self.b = None
        self.psum = 0
        self.active_cycles = 0
        self.next_a = None
        self.next_b = None

    def step(self):
        if self.a is not None and self.b is not None:
            self.psum += self.a * self.b
            self.active_cycles += 1


def os_build_array(M, N):
    """M rows x N cols of PEs. C is M x N."""
    return [[OS_PE() for _ in range(N)] for _ in range(M)]


def os_step_array(array):
    M = len(array)
    N = len(array[0])

    for i in range(M):
        for j in range(N):
            array[i][j].step()

    for i in range(M):
        for j in range(N):
            pe = array[i][j]
            pe.next_a = pe.a
            pe.next_b = pe.b

    for i in range(M):
        for j in range(N):
            array[i][j].a = None
            array[i][j].b = None

    for i in range(M):
        for j in range(N):
            pe = array[i][j]
            if pe.next_a is not None and j + 1 < N:
                array[i][j + 1].a = pe.next_a
            if pe.next_b is not None and i + 1 < M:
                array[i + 1][j].b = pe.next_b


def os_inject_inputs(array, A, B, t, stats):
    M = len(array)
    N = len(array[0])
    K = len(A[0])

    for i in range(M):
        k = t - i
        if 0 <= k < K:
            array[i][0].a = A[i][k]
            stats["A_in"] += 1

    for j in range(N):
        k = t - j
        if 0 <= k < len(B):
            array[0][j].b = B[k][j]
            stats["B_in"] += 1


def os_run(A, B):
    """A: MxK, B: KxN -> C: MxN, using an MxN output-stationary array."""
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    assert len(B) == K, "Inner dimensions must match"

    array = os_build_array(M, N)

    # Tight-ish bound: fill latency (M-1)+(N-1) rows/cols of skew,
    # plus K cycles of streaming, plus 1 for the final compute.
    total_cycles = (M - 1) + (N - 1) + K + 1

    stats = {"A_in": 0, "B_in": 0}
    for t in range(total_cycles):
        os_inject_inputs(array, A, B, t, stats)
        os_step_array(array)

    C = [[array[i][j].psum for j in range(N)] for i in range(M)]
    return C, total_cycles, stats, array


def os_utilization(array, cycles):
    total_active = sum(pe.active_cycles for row in array for pe in row)
    num_pes = len(array) * len(array[0])
    return total_active / (num_pes * cycles), total_active


# ===========================================================
# 2) WEIGHT-STATIONARY DATAFLOW
# ===========================================================
class WS_PE:
    __slots__ = ("weight", "a", "psum_in", "active_cycles", "next_a", "next_psum")

    def __init__(self, weight):
        self.weight = weight
        self.a = None
        self.psum_in = None
        self.active_cycles = 0
        self.next_a = None
        self.next_psum = None

    def step(self):
        if self.a is not None and self.psum_in is not None:
            self.next_psum = self.psum_in + self.a * self.weight
            self.next_a = self.a
            self.active_cycles += 1
        else:
            self.next_psum = None
            self.next_a = None


def ws_build_array(B):
    """Array is K x N (matches weight matrix B: KxN), one weight per PE."""
    K = len(B)
    N = len(B[0])
    return [[WS_PE(B[k][n]) for n in range(N)] for k in range(K)]


def ws_step_array(array, outputs, t):
    """Advance one cycle: compute, latch, shift a right / psum down,
    and capture values exiting the bottom row into `outputs`."""
    K = len(array)
    N = len(array[0])

    for k in range(K):
        for n in range(N):
            array[k][n].step()

    # capture exits from the bottom row before we overwrite state
    for n in range(N):
        exiting = array[K - 1][n].next_psum
        if exiting is not None:
            outputs[n].append(exiting)

    # shift: activation moves right (same row), psum moves down (same col)
    new_a = [[None] * N for _ in range(K)]
    new_psum = [[None] * N for _ in range(K)]

    for k in range(K):
        for n in range(N):
            pe = array[k][n]
            if pe.next_a is not None and n + 1 < N:
                new_a[k][n + 1] = pe.next_a
            if pe.next_psum is not None and k + 1 < K:
                new_psum[k + 1][n] = pe.next_psum

    for k in range(K):
        for n in range(N):
            array[k][n].a = new_a[k][n]
            array[k][n].psum_in = new_psum[k][n]


def ws_inject_inputs(array, A, t, stats):
    """Inject activations (skewed by row k) and zero partial sums
    (skewed by column n) for each new output row m."""
    K = len(array)
    N = len(array[0])
    M = len(A)

    # activation A[m][k] enters row k, col 0, when k = t - m
    for m in range(M):
        k = t - m
        if 0 <= k < K:
            array[k][0].a = A[m][k]
            stats["A_in"] += 1

    # psum = 0 enters row 0, col n, when n = t - m  =>  m = t - n
    for n in range(N):
        m = t - n
        if 0 <= m < M:
            array[0][n].psum_in = 0


def ws_run(A, B):
    """A: MxK, B: KxN -> C: MxN, using a KxN weight-stationary array."""
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    assert len(B) == K, "Inner dimensions must match"

    array = ws_build_array(B)
    outputs = [[] for _ in range(N)]  # values exiting each column, in order

    total_cycles = (M - 1) + (K - 1) + N + 1  # fill + drain + margin

    stats = {"A_in": 0}
    for t in range(total_cycles):
        ws_inject_inputs(array, A, t, stats)
        ws_step_array(array, outputs, t)

    # each column n's outputs arrive in increasing m order
    C = [[0 for _ in range(N)] for _ in range(M)]
    for n in range(N):
        col_vals = outputs[n][:M]
        for m, val in enumerate(col_vals):
            C[m][n] = val

    return C, total_cycles, stats, array


def ws_utilization(array, cycles):
    total_active = sum(pe.active_cycles for row in array for pe in row)
    num_pes = len(array) * len(array[0])
    return total_active / (num_pes * cycles), total_active


# ===========================================================
# 3) ENERGY / BANDWIDTH MODEL
# ===========================================================
# Rough relative energy costs per access, in units of "1 MAC energy",
# loosely following the orders of magnitude reported in Eyeriss-style
# analyses (DRAM >> SRAM/NoC >> MAC). These are illustrative, not
# vendor-specific numbers.
E_MAC = 1
E_SRAM = 6
E_DRAM = 200


def os_energy_estimate(A_in, B_in, num_macs):
    """Output-stationary: A and B both read from DRAM once per element
    as they stream in; MACs happen locally at each PE (no psum traffic
    since psum never leaves its PE until the end)."""
    dram_energy = (A_in + B_in) * E_DRAM
    mac_energy = num_macs * E_MAC
    return dram_energy + mac_energy, dram_energy, mac_energy


def ws_energy_estimate(A_in, num_macs, num_weights):
    """Weight-stationary: weights loaded once from DRAM (amortized over
    all reuse), activations read from DRAM as they stream in, partial
    sums move through on-chip SRAM/NoC as they accumulate down columns."""
    dram_energy = (A_in + num_weights) * E_DRAM
    # each MAC produces one psum hop to the next PE (except the last row)
    sram_energy = num_macs * E_SRAM
    mac_energy = num_macs * E_MAC
    return dram_energy + sram_energy + mac_energy, dram_energy, sram_energy, mac_energy


# ===========================================================
# 4) CORRECTNESS TESTS
# ===========================================================
def run_correctness_tests(num_tests=25, seed=0):
    random.seed(seed)
    failures = []

    for trial in range(num_tests):
        M = random.randint(1, 5)
        K = random.randint(1, 5)
        N = random.randint(1, 5)

        A = random_matrix(M, K)
        B = random_matrix(K, N)
        C_ref = matmul_ref(A, B)

        C_os, _, _, _ = os_run(A, B)
        C_ws, _, _, _ = ws_run(A, B)

        if C_os != C_ref:
            failures.append(("OS", M, K, N, A, B, C_os, C_ref))
        if C_ws != C_ref:
            failures.append(("WS", M, K, N, A, B, C_ws, C_ref))

    if failures:
        print(f"FAILED {len(failures)} / {num_tests * 2} checks:")
        for f in failures[:5]:
            mode, M, K, N, A, B, got, ref = f
            print(f"  [{mode}] shape M={M},K={K},N={N}")
            print(f"    A={A}\n    B={B}\n    got={got}\n    ref={ref}")
    else:
        print(f"All {num_tests * 2} correctness checks passed "
              f"(random shapes, {num_tests} trials each dataflow).")

    return len(failures) == 0


# ===========================================================
# 5) DEMO + GRAPH EXPERIMENTS
# ===========================================================
def run_graph_experiments():
    import matplotlib.pyplot as plt

    # -------------------------
    # Graph 1: utilization vs array size (N), square, OS vs WS
    # -------------------------
    K = 16
    Ns = [2, 3, 4, 5, 6]

    util_os, util_ws = [], []
    for N in Ns:
        A = [[1 for _ in range(K)] for _ in range(N)]
        B = [[1 for _ in range(N)] for _ in range(K)]

        C_os, cyc_os, _, arr_os = os_run(A, B)
        u_os, _ = os_utilization(arr_os, cyc_os)
        util_os.append(u_os)

        C_ws, cyc_ws, _, arr_ws = ws_run(A, B)
        u_ws, _ = ws_utilization(arr_ws, cyc_ws)
        util_ws.append(u_ws)

    plt.figure()
    plt.plot(Ns, util_os, marker='o', label="Output-stationary")
    plt.plot(Ns, util_ws, marker='s', label="Weight-stationary")
    plt.xlabel("Array size (N x N)")
    plt.ylabel("Average PE utilization")
    plt.title("Utilization vs array size")
    plt.legend()
    plt.grid(True)
    plt.savefig("/kaggle/working/graph1_utilization_vs_N.png", dpi=120)
    plt.close()

    # -------------------------
    # Graph 2: utilization vs matrix size (K), fixed 4x4 array
    # -------------------------
    N = 4
    Ks = [4, 8, 16, 32, 64]

    util_os_k, util_ws_k = [], []
    for K in Ks:
        A = [[1 for _ in range(K)] for _ in range(N)]
        B = [[1 for _ in range(N)] for _ in range(K)]

        C_os, cyc_os, _, arr_os = os_run(A, B)
        u_os, _ = os_utilization(arr_os, cyc_os)
        util_os_k.append(u_os)

        C_ws, cyc_ws, _, arr_ws = ws_run(A, B)
        u_ws, _ = ws_utilization(arr_ws, cyc_ws)
        util_ws_k.append(u_ws)

    plt.figure()
    plt.plot(Ks, util_os_k, marker='o', label="Output-stationary")
    plt.plot(Ks, util_ws_k, marker='s', label="Weight-stationary")
    plt.xlabel("Matrix inner dimension (K)")
    plt.ylabel("Average PE utilization")
    plt.title("Utilization vs matrix size (fixed 4x4 array)")
    plt.legend()
    plt.grid(True)
    plt.savefig("/kaggle/working/graph2_utilization_vs_K.png", dpi=120)
    plt.close()

    # -------------------------
    # Graph 3: energy estimate vs K, OS vs WS (fixed 4x4 array)
    # -------------------------
    N = 4
    Ks = [4, 8, 16, 32, 64]
    energy_os, energy_ws = [], []

    for K in Ks:
        A = [[1 for _ in range(K)] for _ in range(N)]
        B = [[1 for _ in range(N)] for _ in range(K)]

        C_os, cyc_os, stats_os, arr_os = os_run(A, B)
        num_macs_os = sum(pe.active_cycles for row in arr_os for pe in row)
        total_e_os, _, _ = os_energy_estimate(stats_os["A_in"], stats_os["B_in"], num_macs_os)
        energy_os.append(total_e_os)

        C_ws, cyc_ws, stats_ws, arr_ws = ws_run(A, B)
        num_macs_ws = sum(pe.active_cycles for row in arr_ws for pe in row)
        num_weights = N * K
        total_e_ws, _, _, _ = ws_energy_estimate(stats_ws["A_in"], num_macs_ws, num_weights)
        energy_ws.append(total_e_ws)

    plt.figure()
    plt.plot(Ks, energy_os, marker='o', label="Output-stationary")
    plt.plot(Ks, energy_ws, marker='s', label="Weight-stationary")
    plt.xlabel("Matrix inner dimension (K)")
    plt.ylabel("Estimated relative energy (a.u.)")
    plt.title("Energy estimate vs matrix size (fixed 4x4 array)")
    plt.legend()
    plt.grid(True)
    plt.savefig("/kaggle/working/graph3_energy_vs_K.png", dpi=120)
    plt.close()

    print("Saved graph1_utilization_vs_N.png, graph2_utilization_vs_K.png, "
          "graph3_energy_vs_K.png")


if __name__ == "__main__":
    # ---- correctness tests (including rectangular shapes) ----
    ok = run_correctness_tests(num_tests=25, seed=0)
    print()

    # ---- worked demo (rectangular, non-trivial matrices) ----
    A = [
        [1, 2, 3],
        [4, 5, 6],
    ]  # 2x3
    B = [
        [7, 8],
        [9, 10],
        [11, 12],
    ]  # 3x2

    C_os, cyc_os, stats_os, arr_os = os_run(A, B)
    C_ws, cyc_ws, stats_ws, arr_ws = ws_run(A, B)
    C_ref = matmul_ref(A, B)

    print("Reference output:      ", C_ref)
    print("Output-stationary:      ", C_os, f"(cycles={cyc_os})")
    print("Weight-stationary:      ", C_ws, f"(cycles={cyc_ws})")
    assert C_os == C_ref, "Output-stationary mismatch!"
    assert C_ws == C_ref, "Weight-stationary mismatch!"
    print("Both dataflows match the reference on a rectangular (2x3)*(3x2) case.\n")

    u_os, active_os = os_utilization(arr_os, cyc_os)
    u_ws, active_ws = ws_utilization(arr_ws, cyc_ws)
    print(f"OS utilization: {u_os:.3f}  (active PE-cycles={active_os})")
    print(f"WS utilization: {u_ws:.3f}  (active PE-cycles={active_ws})")

    if ok:
        run_graph_experiments()


from IPython.display import Image, display
display(Image("/kaggle/working/graph1_utilization_vs_N.png"))
display(Image("/kaggle/working/graph2_utilization_vs_K.png"))
display(Image("/kaggle/working/graph3_energy_vs_K.png"))
