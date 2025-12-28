import pyperf


def add_python_benchmarks(runner: pyperf.Runner) -> None:
    """
    Pure-Python microbenchmarks.

    Keep these simple and allocation-light in the timed region so results are more stable.
    """

    # Tight integer loop (measures loop + integer arithmetic)
    def int_loop_add_1e5() -> int:
        s = 0
        for i in range(100_000):
            s += i
        return s

    runner.bench_func("python.int_loop_add[1e5]", int_loop_add_1e5, inner_loops=20)

    # Floating point arithmetic (keeps values in registers; low allocation)
    def float_mul_add_1e5() -> float:
        x = 1.0001
        y = 0.9999
        for _ in range(100_000):
            x = x * y + 0.000001
        return x

    runner.bench_func("python.float_mul_add[1e5]", float_mul_add_1e5, inner_loops=10)

    # List append (allocation happens as list grows; keep count fixed and cache append)
    def list_append_5e4() -> int:
        lst: list[int] = []
        append = lst.append
        for i in range(50_000):
            append(i)
        return len(lst)

    runner.bench_func("python.list_append[5e4]", list_append_5e4, inner_loops=30)

    # Dict lookup (hot loop; dict is pre-built)
    d = {i: i + 1 for i in range(10_000)}

    def dict_get_hit_1e5() -> int:
        s = 0
        get = d.get
        for i in range(100_000):
            s += get(i % 10_000, 0)
        return s

    runner.bench_func("python.dict_get_hit[1e5]", dict_get_hit_1e5, inner_loops=30)

    # Function call overhead (small Python call in a loop)
    def _tiny(x: int) -> int:
        return x + 1

    def function_calls_2e5() -> int:
        s = 0
        f = _tiny
        for i in range(200_000):
            s += f(i)
        return s

    runner.bench_func("python.function_calls[2e5]", function_calls_2e5, inner_loops=10)

    # Attribute access (common Python overhead)
    class C:
        __slots__ = ("x",)

        def __init__(self) -> None:
            self.x = 1

    c = C()

    def attr_get_2e5() -> int:
        s = 0
        obj = c
        for _ in range(200_000):
            s += obj.x
        return s

    runner.bench_func("python.attr_get[2e5]", attr_get_2e5, inner_loops=20)


