import jax
import jax.numpy as jnp


def main() -> int:
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    y = jax.jit(lambda a: a * 2.0)(x)
    y.block_until_ready()
    print(f"candidate=jax version={jax.__version__}")
    print(f"devices={jax.devices()}")
    print(f"result={list(map(float, y.tolist()))}")
    assert y.devices(), "JAX result is not on a device"
    assert y.tolist() == [2.0, 4.0, 6.0, 8.0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
