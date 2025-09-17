# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Re-export the actual implementations so that
# `from ..layers import MemEffAttention, Mlp` works without changing model code.

# Export MemEffAttention
try:
    from .mem_efficient_attention import MemEffAttention  # typical filename
except Exception:
    try:
        from .attention import MemEffAttention  # alternative filename in some forks
    except Exception:
        # If a fork uses a slightly different class name, alias it.
        from .mem_efficient_attention import (
            MemEfficientAttention as MemEffAttention,  # type: ignore
        )

# Export Mlp
try:
    from .mlp import Mlp
except Exception:
    from .feedforward import Mlp  # fallback if a fork uses this name

__all__ = ["MemEffAttention", "Mlp"]


# Export NestedTensorBlock so that
# `from ..layers import NestedTensorBlock as Block` works.
try:
    from .nested_tensor_block import NestedTensorBlock
except Exception:
    try:
        # Some forks name the file/class simply "block"
        from .block import NestedTensorBlock
    except Exception:
        try:
            # Some forks expose it as "Block" — alias it to NestedTensorBlock
            from .nested_tensor_block import Block as NestedTensorBlock  # type: ignore
        except Exception:
            # Fallback: a "transformer_block.py" with class "Block"
            from .transformer_block import Block as NestedTensorBlock  # type: ignore

# (Optional) Export DropPath if the model expects it via layers namespace
try:
    pass
except Exception:
    pass

# Update the package export list
try:
    __all__
except NameError:
    __all__ = []
for _n in ["MemEffAttention", "Mlp", "NestedTensorBlock", "DropPath"]:
    if _n not in __all__:
        __all__.append(_n)

# Export PatchEmbed (various repos place it in different files)
try:
    from .patch_embed import PatchEmbed
except Exception:
    try:
        from .embedding import PatchEmbed
    except Exception:
        try:
            from .patch import PatchEmbed  # some forks
        except Exception:
            pass  # leave undefined if truly absent

# Export SwiGLUFFNFused (and compatible fallbacks)
# Prefer the exact file we detected in this repo: swiglu_ffn.py
try:
    from .swiglu_ffn import SwiGLUFFNFused
except Exception:
    try:
        # Some repos put fused/non-fused in swiglu.py
        from .swiglu import SwiGLUFFNFused  # type: ignore
    except Exception:
        try:
            # Another possible filename
            from .ffn_fused import SwiGLUFFNFused  # type: ignore
        except Exception:
            try:
                from .fused_swiglu import SwiGLUFFNFused  # type: ignore
            except Exception:
                try:
                    from .swiglu_fused import SwiGLUFFNFused  # type: ignore
                except Exception:
                    try:
                        from .fused import SwiGLUFFNFused  # type: ignore
                    except Exception:
                        # Final non-fused fallback: alias to keep code running (may be slightly slower)
                        try:
                            from .swiglu_ffn import (
                                SwiGLUFFN as SwiGLUFFNFused,  # type: ignore
                            )
                        except Exception:
                            try:
                                from .swiglu import (
                                    SwiGLUFFN as SwiGLUFFNFused,  # type: ignore
                                )
                            except Exception:
                                pass

# Update __all__ if symbol is now available
if "SwiGLUFFNFused" in globals():
    try:
        __all__
    except NameError:
        __all__ = []
    if "SwiGLUFFNFused" not in __all__:
        __all__.append("SwiGLUFFNFused")


# (Optional) Common extras that vision_transformer.py may import later.
# They do not change behavior; only make package-level imports succeed.
try:
    from .norms import LayerNorm2d  # if present
except Exception:
    try:
        from .layer_norm_2d import LayerNorm2d  # alt filename
    except Exception:
        pass

try:
    pass
except Exception:
    try:
        pass
    except Exception:
        pass

try:
    pass
except Exception:
    pass

try:
    pass
except Exception:
    pass

try:
    pass
except Exception:
    pass

# Keep __all__ updated
try:
    __all__
except NameError:
    __all__ = []
for _n in [
    "MemEffAttention",
    "Mlp",
    "NestedTensorBlock",
    "DropPath",
    "PatchEmbed",
    "SwiGLUFFNFused",
    "LayerNorm2d",
    "LayerScale",
    "RelativePositionBias",
    "get_2d_sincos_pos_embed",
    "trunc_normal_",
]:
    if _n not in __all__ and _n in globals():
        __all__.append(_n)
