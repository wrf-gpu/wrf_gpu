"""Saturation helpers transcribed from WRF Thompson polynomial fits."""

from __future__ import annotations

import jax.numpy as jnp


def latent_heat_vaporization(T):
    """Encapsulates WRF's temperature-dependent lvap formula reused in updates."""

    return 2.5e6 + (2106.0 - 4218.0) * (T - 273.15)


def cp_inverse(qv):
    """Encapsulates WRF's moist heat-capacity reciprocal used by temperature tendencies."""

    return 1.0 / (1004.0 * (1.0 + 0.887 * qv))


def saturation_mixing_ratio_liquid(p, T):
    """Evaluates WRF RSLF polynomial from module_mp_thompson.F.pre lines 5444-5468."""

    x = jnp.maximum(-80.0, T - 273.16)
    esl = (
        0.611583699e03
        + x
        * (
            0.444606896e02
            + x
            * (
                0.143177157e01
                + x
                * (
                    0.264224321e-1
                    + x
                    * (
                        0.299291081e-3
                        + x
                        * (
                            0.203154182e-5
                            + x * (0.702620698e-8 + x * (0.379534310e-11 + x * -0.321582393e-13))
                        )
                    )
                )
            )
        )
    )
    esl = jnp.minimum(esl, p * 0.15)
    return 0.622 * esl / (p - esl)


def saturation_mixing_ratio_ice(p, T):
    """Evaluates WRF RSIF polynomial from module_mp_thompson.F.pre lines 5473-5490."""

    x = jnp.maximum(-80.0, T - 273.16)
    esi = (
        0.609868993e03
        + x
        * (
            0.499320233e02
            + x
            * (
                0.184672631e01
                + x
                * (
                    0.402737184e-1
                    + x
                    * (
                        0.565392987e-3
                        + x
                        * (
                            0.521693933e-5
                            + x * (0.307839583e-7 + x * (0.105785160e-9 + x * 0.161444444e-12))
                        )
                    )
                )
            )
        )
    )
    esi = jnp.minimum(esi, p * 0.15)
    return 0.622 * esi / jnp.maximum(1.0e-4, p - esi)
