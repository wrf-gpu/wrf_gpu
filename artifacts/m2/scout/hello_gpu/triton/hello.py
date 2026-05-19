import torch
import triton
import triton.language as tl


@triton.jit
def times_two(x_ptr, y_ptr, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, values * 2.0, mask=mask)


def main() -> int:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    times_two[(1,)](x, y, x.numel(), block=8)
    torch.cuda.synchronize()
    result = y.cpu().tolist()
    device = torch.cuda.get_device_name(0)
    print(f"candidate=triton version={triton.__version__}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"device={device}")
    print(f"result={result}")
    assert result == [2.0, 4.0, 6.0, 8.0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
