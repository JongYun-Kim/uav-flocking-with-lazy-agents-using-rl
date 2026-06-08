"""Global RNG seeding and strict determinism for reproducible RLlib 2.1.0 runs.

Public API:

* ``set_global_seed(seed)`` -- seed random/numpy/torch RNGs and set deterministic-friendly
  env / cuDNN flags. Call it on the DRIVER (before ``tune.run``). Does NOT enable
  ``torch.use_deterministic_algorithms`` (that is handled by the RLlib patch below, in the
  trainer process, at the right time).

* ``enable_strict_determinism()`` -- re-assert valid cuBLAS config + enable
  ``use_deterministic_algorithms`` in the CURRENT process. Used as a belt-and-suspenders call
  from an RLlib ``on_algorithm_init`` callback.

* ``patch_rllib_determinism()`` -- wrap RLlib's ``update_global_seed_if_necessary`` so that, in
  every training process, it emits a VALID ``CUBLAS_WORKSPACE_CONFIG`` and enables strict mode.
  Applied automatically on import (see bottom of module).

Why a patch is necessary (RLlib 2.1.0 / torch cu113 specifics):
  RLlib's ``update_global_seed_if_necessary`` runs early in ``Algorithm.setup()`` in every
  training process. On CUDA>=10.2 builds it (a) seeds random/numpy/torch + sets
  ``cudnn.deterministic`` but SKIPS ``torch.use_deterministic_algorithms``, and (b) writes the
  MALFORMED ``CUBLAS_WORKSPACE_CONFIG="4096:8"`` (torch's determinism check requires the
  leading colon: ":4096:8" or ":16:8"). If we enable ``use_deterministic_algorithms`` with that
  malformed value present, the first GPU cuBLAS matmul raises RuntimeError. RLlib's seed call
  runs *before* the policy is built (before the first cuBLAS matmul / handle creation), which is
  exactly where the value must be corrected -- fixing it later (e.g. in ``on_algorithm_init``,
  which runs after the policy's dummy-batch matmul) is too late because the cuBLAS handle has
  already been created with the bad value. So we wrap RLlib's own seed call.

Safe to import on CPU-only / torch-absent / ray-absent environments.
"""
import os
import random

import numpy as np

# torch's deterministic cuBLAS check accepts ONLY these exact strings (leading colon required).
VALID_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_global_seed(seed: int) -> int:
    """Seed RNGs and set deterministic-friendly env/cuDNN flags (no strict-algo enforcement).

    Enables reproducibility of all RNG-driven randomness (numpy in the env, torch weight init /
    action sampling) and sets a VALID ``CUBLAS_WORKSPACE_CONFIG`` so child processes (Ray
    actors/workers) inherit it. Does NOT call ``torch.use_deterministic_algorithms`` -- the RLlib
    patch enables that inside each trainer process at the correct time (see module docstring).

    Returns the seed (convenience for logging).
    """
    # PYTHONHASHSEED set here only affects child/forked processes spawned after this point
    # (e.g. Ray workers); it cannot retroactively change the running interpreter's hashing.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = VALID_CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def enable_strict_determinism(warn_only: bool = False) -> None:
    """Re-assert valid cuBLAS config + enable strict torch determinism in the CURRENT process.

    Belt-and-suspenders for an RLlib ``on_algorithm_init`` callback. The heavy lifting is done by
    :func:`patch_rllib_determinism` (which runs at the correct, earlier time); this call is a
    harmless, idempotent re-assertion. ``warn_only`` is forwarded to
    ``torch.use_deterministic_algorithms`` (only downgrades "no deterministic implementation"
    errors; it does not affect the cuBLAS-config error, which the env re-assertion handles).
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = VALID_CUBLAS_WORKSPACE_CONFIG
    try:
        import torch
    except ImportError:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


_RLLIB_PATCHED = False


def patch_rllib_determinism() -> None:
    """Wrap RLlib 2.1.0's ``update_global_seed_if_necessary`` to fix its two GPU-determinism
    deficiencies at the right time (it runs early in ``Algorithm.setup()``, before the policy is
    built / before the first GPU cuBLAS matmul):

      1. Repair the malformed ``CUBLAS_WORKSPACE_CONFIG="4096:8"`` -> valid ":4096:8".
      2. Enable ``torch.use_deterministic_algorithms(True)`` (RLlib skips it on CUDA>=10.2).

    Idempotent. Applied automatically on import of this module so that it is installed before any
    ``Algorithm.setup()`` runs -- including in ``tune.run`` trial actors, which import this module
    (to resolve the ``on_algorithm_init`` callback's reference to ``enable_strict_determinism``)
    before their setup begins.
    """
    global _RLLIB_PATCHED
    if _RLLIB_PATCHED:
        return
    import ray.rllib.algorithms.algorithm as algo_mod
    import ray.rllib.utils.debug.deterministic as det_mod

    original = det_mod.update_global_seed_if_necessary

    def patched(framework=None, seed=None):
        result = original(framework, seed)
        if seed is not None:
            # Repair RLlib's malformed cuBLAS value and enable strict mode, before any cuBLAS
            # handle is created in this process.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = VALID_CUBLAS_WORKSPACE_CONFIG
            if framework == "torch":
                try:
                    import torch
                except ImportError:
                    return result
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
        return result

    det_mod.update_global_seed_if_necessary = patched
    # algorithm.py did `from ...debug import update_global_seed_if_necessary`, so it holds its own
    # module-level reference that must be rebound too.
    algo_mod.update_global_seed_if_necessary = patched
    _RLLIB_PATCHED = True


# Install the patch as a side effect of importing this module, so it is active in every process
# that uses the seeding utilities -- crucially, inside tune.run trial actors before their
# Algorithm.setup() runs. Guarded so importing this module never fails when RLlib is absent.
try:
    patch_rllib_determinism()
except Exception:
    pass
