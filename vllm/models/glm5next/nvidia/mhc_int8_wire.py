# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-scaled 8-bit wire codec for the GLM mHC ownership collectives.

The mHC prefill ownership path exchanges BF16 ``[rows, 4096]`` tensors
through one reduce-scatter and one all-gather per layer. On four DGX Sparks
those collectives are wire-bound at 16 MiB per rank. This module encodes each
row as 32 blocks of 128 one-byte values with one FP32 scale per block, 132
bytes per 128 values instead of 256, so the same collectives move about half
the bytes. Two codecs share the layout: symmetric INT8 (``scale = amax / 127``)
and E4M3 float8 (``scale = amax / 448``); both use a unit scale for
non-positive or NaN block maxima, matching the b12x PCIe DMA codecs.

- all-gather: quantize the owned rows, gather the packed rows, dequantize the
  full tensor. Every rank rebuilds its own rows from the wire payload too, so
  the gathered tensor is bit-identical across ranks.
- reduce-scatter: quantize the full partial, exchange row quarters with an
  all-to-all over the PyNccl communicator, and accumulate the four received
  quarters in FP32 in rank order.

Row layout: ``ROW_BYTES = 4096 + 32 * 4``; bytes ``[0, 4096)`` hold the INT8
payload, bytes ``[4096, 4224)`` hold the 32 FP32 scales.
"""

from __future__ import annotations

import torch

HIDDEN = 4096
BLOCK = 128
BLOCKS_PER_ROW = HIDDEN // BLOCK
SCALE_BYTES = BLOCKS_PER_ROW * 4
ROW_BYTES = HIDDEN + SCALE_BYTES
ROW_FLOATS = ROW_BYTES // 4
_INT8_MAX = 127.0
_E4M3_MAX = 448.0
CODECS = ("i8", "e4m3")

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU test environments
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _quantize_rows_kernel(
        x_ptr,
        payload_ptr,
        scales_ptr,
        x_row_stride,
        E4M3: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        ROW_BYTES: tl.constexpr,
        ROW_FLOATS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, HIDDEN)
        x = tl.load(x_ptr + row * x_row_stride + offs).to(tl.float32)
        blocks = tl.reshape(x, (BLOCKS_PER_ROW, BLOCK))
        amax = tl.max(tl.abs(blocks), axis=1)
        if E4M3:
            scale = tl.where(amax > 0.0, amax / 448.0, 1.0)
            q = tl.reshape(blocks * (1.0 / scale)[:, None], (HIDDEN,))
            packed = q.to(tl.float8e4nv).to(tl.int8, bitcast=True)
        else:
            scale = tl.where(amax > 0.0, amax / 127.0, 1.0)
            q = tl.reshape(blocks * (1.0 / scale)[:, None], (HIDDEN,))
            q = tl.minimum(tl.maximum(tl.floor(q + 0.5), -127.0), 127.0)
            packed = q.to(tl.int8)
        tl.store(payload_ptr + row * ROW_BYTES + offs, packed)
        tl.store(
            scales_ptr
            + row * ROW_FLOATS
            + (HIDDEN // 4)
            + tl.arange(0, BLOCKS_PER_ROW),
            scale,
        )

    @triton.jit
    def _dequantize_rows_kernel(
        payload_ptr,
        scales_ptr,
        out_ptr,
        E4M3: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        ROW_BYTES: tl.constexpr,
        ROW_FLOATS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, HIDDEN)
        raw = tl.load(payload_ptr + row * ROW_BYTES + offs)
        if E4M3:
            q = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            q = raw.to(tl.float32)
        scale = tl.load(
            scales_ptr + row * ROW_FLOATS + (HIDDEN // 4) + tl.arange(0, BLOCKS_PER_ROW)
        )
        x = tl.reshape(q, (BLOCKS_PER_ROW, BLOCK)) * scale[:, None]
        tl.store(
            out_ptr + row * HIDDEN + offs, tl.reshape(x, (HIDDEN,)).to(tl.bfloat16)
        )

    @triton.jit
    def _accumulate_rows_kernel(
        payload_ptr,
        scales_ptr,
        out_ptr,
        rows,
        E4M3: tl.constexpr,
        SHARDS: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        ROW_BYTES: tl.constexpr,
        ROW_FLOATS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, HIDDEN)
        acc = tl.zeros((BLOCKS_PER_ROW, BLOCK), dtype=tl.float32)
        for shard in tl.static_range(SHARDS):
            base = (shard * rows + row) * ROW_BYTES
            raw = tl.load(payload_ptr + base + offs)
            if E4M3:
                q = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            else:
                q = raw.to(tl.float32)
            scale = tl.load(
                scales_ptr
                + (shard * rows + row) * ROW_FLOATS
                + (HIDDEN // 4)
                + tl.arange(0, BLOCKS_PER_ROW)
            )
            acc += tl.reshape(q, (BLOCKS_PER_ROW, BLOCK)) * scale[:, None]
        tl.store(
            out_ptr + row * HIDDEN + offs, tl.reshape(acc, (HIDDEN,)).to(tl.bfloat16)
        )


def _views(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 payload view and FP32 scale view of a packed ``[..., ROW_BYTES]`` buffer."""
    if packed.dtype != torch.uint8 or packed.shape[-1] != ROW_BYTES:
        raise ValueError("packed rows must be uint8 with ROW_BYTES columns")
    if not packed.is_contiguous():
        raise ValueError("packed rows must be contiguous")
    return packed.view(torch.int8), packed.view(torch.float32)


def quantize_rows(
    x: torch.Tensor, out: torch.Tensor | None = None, codec: str = "i8"
) -> torch.Tensor:
    """Encode BF16 ``[rows, 4096]`` into packed uint8 ``[rows, ROW_BYTES]``."""
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != HIDDEN:
        raise ValueError("quantize_rows expects BF16 [rows, 4096]")
    if x.stride(1) != 1:
        raise ValueError("quantize_rows expects unit column stride")
    rows = x.shape[0]
    if out is None:
        out = torch.empty((rows, ROW_BYTES), dtype=torch.uint8, device=x.device)
    elif tuple(out.shape) != (rows, ROW_BYTES):
        raise ValueError("quantize_rows output has the wrong shape")
    if rows == 0:
        return out
    _check_codec(codec)
    if x.is_cuda and _HAS_TRITON:
        payload, scales = _views(out)
        _quantize_rows_kernel[(rows,)](
            x,
            payload,
            scales,
            x.stride(0),
            E4M3=codec == "e4m3",
            HIDDEN=HIDDEN,
            BLOCK=BLOCK,
            BLOCKS_PER_ROW=BLOCKS_PER_ROW,
            ROW_BYTES=ROW_BYTES,
            ROW_FLOATS=ROW_FLOATS,
        )
        return out
    return _quantize_rows_reference(x, out, codec)


def dequantize_rows(
    packed: torch.Tensor, out: torch.Tensor | None = None, codec: str = "i8"
) -> torch.Tensor:
    """Decode packed uint8 ``[rows, ROW_BYTES]`` into BF16 ``[rows, 4096]``."""
    payload, scales = _views(packed)
    rows = packed.shape[0]
    if out is None:
        out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=packed.device)
    elif (
        tuple(out.shape) != (rows, HIDDEN)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise ValueError("dequantize_rows output must be contiguous BF16 [rows, 4096]")
    if rows == 0:
        return out
    _check_codec(codec)
    if packed.is_cuda and _HAS_TRITON:
        _dequantize_rows_kernel[(rows,)](
            payload,
            scales,
            out,
            E4M3=codec == "e4m3",
            HIDDEN=HIDDEN,
            BLOCK=BLOCK,
            BLOCKS_PER_ROW=BLOCKS_PER_ROW,
            ROW_BYTES=ROW_BYTES,
            ROW_FLOATS=ROW_FLOATS,
        )
        return out
    out.copy_(_dequantize_rows_reference(packed, codec=codec))
    return out


def accumulate_rows(
    packed: torch.Tensor, out: torch.Tensor | None = None, codec: str = "i8"
) -> torch.Tensor:
    """Sum packed ``[shards, rows, ROW_BYTES]`` quarters in FP32 into BF16 rows.

    The accumulation order is the shard order, so every caller that receives
    the same shards in the same order produces identical output.
    """
    if packed.ndim != 3:
        raise ValueError("accumulate_rows expects [shards, rows, ROW_BYTES]")
    payload, scales = _views(packed)
    shards, rows = packed.shape[0], packed.shape[1]
    if out is None:
        out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=packed.device)
    elif (
        tuple(out.shape) != (rows, HIDDEN)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise ValueError("accumulate_rows output must be contiguous BF16 [rows, 4096]")
    if rows == 0:
        return out
    _check_codec(codec)
    if packed.is_cuda and _HAS_TRITON:
        _accumulate_rows_kernel[(rows,)](
            payload,
            scales,
            out,
            rows,
            E4M3=codec == "e4m3",
            SHARDS=shards,
            HIDDEN=HIDDEN,
            BLOCK=BLOCK,
            BLOCKS_PER_ROW=BLOCKS_PER_ROW,
            ROW_BYTES=ROW_BYTES,
            ROW_FLOATS=ROW_FLOATS,
        )
        return out
    acc = torch.zeros((rows, HIDDEN), dtype=torch.float32, device=packed.device)
    for shard in range(shards):
        acc += _dequantize_rows_reference(
            packed[shard], dtype=torch.float32, codec=codec
        )
    out.copy_(acc.to(torch.bfloat16))
    return out


def _check_codec(codec: str) -> None:
    if codec not in CODECS:
        raise ValueError(f"unknown wire codec {codec!r}; expected one of {CODECS}")


def _quantize_rows_reference(
    x: torch.Tensor, out: torch.Tensor, codec: str = "i8"
) -> torch.Tensor:
    """Torch reference of the codecs with the kernels' rounding rules."""
    _check_codec(codec)
    rows = x.shape[0]
    blocks = x.float().reshape(rows, BLOCKS_PER_ROW, BLOCK)
    amax = blocks.abs().amax(dim=-1)
    maximum = _E4M3_MAX if codec == "e4m3" else _INT8_MAX
    scale = torch.where(amax > 0, amax / maximum, torch.ones_like(amax))
    q = blocks / scale[..., None]
    payload, scales = _views(out)
    if codec == "e4m3":
        encoded = q.reshape(rows, HIDDEN).to(torch.float8_e4m3fn).view(torch.int8)
    else:
        encoded = (
            torch.floor(q + 0.5)
            .clamp_(-_INT8_MAX, _INT8_MAX)
            .reshape(rows, HIDDEN)
            .to(torch.int8)
        )
    payload[:, :HIDDEN].copy_(encoded)
    scales[:, HIDDEN // 4 :].copy_(scale)
    return out


def _dequantize_rows_reference(
    packed: torch.Tensor, dtype: torch.dtype = torch.bfloat16, codec: str = "i8"
) -> torch.Tensor:
    _check_codec(codec)
    payload, scales = _views(packed)
    rows = packed.shape[0]
    raw = payload[:, :HIDDEN]
    values = raw.view(torch.float8_e4m3fn).float() if codec == "e4m3" else raw.float()
    q = values.reshape(rows, BLOCKS_PER_ROW, BLOCK)
    scale = scales[:, HIDDEN // 4 :]
    return (q * scale[..., None]).reshape(rows, HIDDEN).to(dtype)
