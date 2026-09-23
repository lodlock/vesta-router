"""Training, evaluation and release tooling for Vesta's local tool router.

Nothing in this package trains a model or talks to a network. It is the part of
the pipeline that can be right or wrong without a GPU: corpus validation, the
state-block renderer, manifest validation and the promotion gates.

The contract it implements lives in the Vesta repository at
``docs/router-model-contract.md``, and CONTRACT.md here records which side owns
what.
"""

__all__ = ["__version__"]

# The tooling's version, not any model's. Model versions are parsed by
# `vesta_router.version` and have their own scheme.
__version__ = "0.1.0"
