import cupy as cp
import dace
import gt4py


N = 4


@dace.program
def times_two(a: dace.float32[N], b: dace.float32[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0


def main() -> int:
    sdfg = times_two.to_sdfg()
    sdfg.apply_gpu_transformations()
    a = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
    b = cp.zeros_like(a)
    sdfg(A=a, B=b)
    cp.cuda.Stream.null.synchronize()
    result = b.get().tolist()
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"candidate=gt4py gt4py_version={gt4py.__version__} dace_version={dace.__version__}")
    print(f"device={device}")
    print(f"result={result}")
    assert result == [2.0, 4.0, 6.0, 8.0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
