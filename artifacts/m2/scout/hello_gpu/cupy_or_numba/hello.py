import cupy as cp


def main() -> int:
    x = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
    y = x * 2.0
    cp.cuda.Stream.null.synchronize()
    result = y.get().tolist()
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"candidate=cupy_or_numba implementation=cupy version={cp.__version__}")
    print(f"device={device}")
    print(f"result={result}")
    assert result == [2.0, 4.0, 6.0, 8.0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
