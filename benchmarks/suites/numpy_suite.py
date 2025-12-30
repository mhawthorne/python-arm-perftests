import numpy as np
import pyperf


def add_numpy_benchmarks(runner: pyperf.Runner) -> None:
    rng = np.random.default_rng(0)

    # 1D dot product (BLAS-backed on many builds)
    a1 = rng.standard_normal(2**22, dtype=np.float64)
    b1 = rng.standard_normal(2**22, dtype=np.float64)

    def dot_1d_f64() -> float:
        return float(np.dot(a1, b1))

    runner.bench_func("numpy.dot[2^22,f64]", dot_1d_f64, inner_loops=10)

    # Matrix multiply (BLAS-heavy)
    a2 = rng.standard_normal((512, 512), dtype=np.float32)
    b2 = rng.standard_normal((512, 512), dtype=np.float32)
    out2 = np.empty((512, 512), dtype=np.float32)

    def matmul_512_f32() -> float:
        np.matmul(a2, b2, out=out2)
        return float(out2[0, 0])

    runner.bench_func("numpy.matmul[512x512,f32]", matmul_512_f32, inner_loops=5)

    # FFT (PocketFFT inside NumPy; not BLAS)
    a3 = rng.standard_normal(2**17, dtype=np.float64)

    def rfft_2p17_f64() -> float:
        y = np.fft.rfft(a3)
        return float(y.real[0])

    runner.bench_func("numpy.fft.rfft[2^17,f64]", rfft_2p17_f64, inner_loops=3)

    # Elementwise ufunc
    a4 = rng.standard_normal(2**20, dtype=np.float32)
    out4 = np.empty_like(a4)

    def exp_2p20_f32() -> float:
        np.exp(a4, out=out4)
        return float(out4[0])

    runner.bench_func("numpy.exp[2^20,f32]", exp_2p20_f32, inner_loops=10)

    # Reduction
    a5 = rng.standard_normal(2**23, dtype=np.float64)

    def sum_2p23_f64() -> float:
        return float(a5.sum())

    runner.bench_func("numpy.sum[2^23,f64]", sum_2p23_f64, inner_loops=10)


