# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-scaled 8-bit wire codec: bounds, packing and ownership collectives."""

from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch

from vllm.models.glm5next.nvidia import mhc_int8_wire as wire

pytestmark = pytest.mark.cpu_test


def _rows(rows: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, wire.HIDDEN, generator=g) * 3.0
    # Heavy tails like a residual stream: a few large outliers per row.
    idx = torch.randint(0, wire.HIDDEN, (rows, 4), generator=g)
    x.scatter_(1, idx, torch.randn(rows, 4, generator=g) * 60.0)
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("codec", wire.CODECS)
def test_reference_roundtrip_bounds(codec):
    x = _rows(16)
    packed = wire.quantize_rows(x, codec=codec)
    assert packed.shape == (16, wire.ROW_BYTES) and packed.dtype == torch.uint8
    # Bound the codec itself in FP32; the BF16 output adds its own half-ulp.
    back = wire._dequantize_rows_reference(packed, dtype=torch.float32, codec=codec)
    scales = packed.view(torch.float32)[:, wire.HIDDEN // 4 :]
    blocks = x.float().reshape(16, wire.BLOCKS_PER_ROW, wire.BLOCK)
    err = (back.reshape_as(blocks) - blocks).abs()
    if codec == "i8":
        # Symmetric INT8 with round-half-up: error at most half a scale step.
        assert torch.all(err <= scales[..., None] / 2 + 1e-6)
    else:
        # E4M3 keeps three mantissa bits: relative error at most 2^-4 of the
        # value, plus the subnormal floor of the block scale.
        assert torch.all(err <= blocks.abs() * 2**-4 + scales[..., None] * 2**-9 + 1e-6)
    # The block maximum survives exactly for INT8 and within one ulp for E4M3.
    amax = blocks.abs().amax(-1)
    rebuilt = back.reshape_as(blocks).abs().amax(-1)
    assert torch.allclose(
        amax, rebuilt, rtol=2**-4 if codec == "e4m3" else 1e-6, atol=1e-6
    )


@pytest.mark.parametrize("codec", wire.CODECS)
def test_zero_and_nan_blocks_use_unit_scale(codec):
    x = torch.zeros(2, wire.HIDDEN, dtype=torch.bfloat16)
    x[1, : wire.BLOCK] = float("nan")
    packed = wire.quantize_rows(x, codec=codec)
    scales = packed.view(torch.float32)[:, wire.HIDDEN // 4 :]
    assert torch.all(scales == 1.0)
    back = wire.dequantize_rows(packed, codec=codec)
    assert torch.all(back[0] == 0)


def test_accumulate_matches_fp32_sum_of_dequantized_shards():
    shards = torch.stack([wire.quantize_rows(_rows(8, seed=s)) for s in range(4)])
    out = wire.accumulate_rows(shards)
    expected = sum(
        wire._dequantize_rows_reference(shards[s], dtype=torch.float32)
        for s in range(4)
    ).to(torch.bfloat16)
    assert torch.equal(out, expected)


def test_codec_rejects_unknown_names_and_bad_layouts():
    with pytest.raises(ValueError):
        wire.quantize_rows(_rows(2), codec="fp4")
    with pytest.raises(ValueError):
        wire.dequantize_rows(torch.zeros(2, wire.ROW_BYTES - 1, dtype=torch.uint8))
    with pytest.raises(ValueError):
        wire.quantize_rows(torch.zeros(2, wire.HIDDEN))  # float32 input


class FakeComm:
    """Records PyNccl calls; all_gather concatenates, send/recv use a mailbox."""

    available = True
    disabled = False

    def __init__(self, world, rank, mailbox):
        self.device = torch.device("cpu")
        self.world, self.rank, self.mailbox = world, rank, mailbox
        self.calls = []

    def all_gather(self, output, input_, stream=None):
        self.calls.append(("ag", tuple(input_.shape), input_.dtype))
        for r in range(self.world):
            output[r * input_.shape[0] : (r + 1) * input_.shape[0]].copy_(
                self.mailbox["ag"][r]
            )

    def group_start(self):
        self.calls.append(("group_start",))

    def group_end(self):
        self.calls.append(("group_end",))

    def send(self, tensor, dst, stream=None):
        self.calls.append(("send", dst, tuple(tensor.shape)))
        self.mailbox["p2p"][(self.rank, dst)] = tensor.clone()

    def recv(self, tensor, src, stream=None):
        self.calls.append(("recv", src, tuple(tensor.shape)))
        tensor.copy_(self.mailbox["p2p"][(src, self.rank)])


@pytest.mark.parametrize("codec", wire.CODECS)
def test_ownership_all_gather_and_all_to_all_are_rank_identical(codec, monkeypatch):
    from vllm.models.glm5next.nvidia.mhc_prefill_sharding import PrefillOwnership

    monkeypatch.setattr(
        "vllm.utils.torch_utils.current_stream",
        lambda: NS(record_stream=lambda t: None),
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(wire, "_HAS_TRITON", False)  # reference path on CPU
    rows = 32
    owned = [_rows(rows // 4, seed=r) for r in range(4)]
    mailbox: dict[str, Any] = {
        "ag": [wire.quantize_rows(o, codec=codec) for o in owned],
        "p2p": {},
    }
    owners = [
        PrefillOwnership(
            comm=FakeComm(4, r, mailbox),
            rank=r,
            rows=rows,
            wire_codec=codec,
            wire_mode="a2a",
        )
        for r in range(4)
    ]
    gathered = [o.all_gather(owned[o.rank]) for o in owners]
    expected = torch.cat([wire.dequantize_rows(m, codec=codec) for m in mailbox["ag"]])
    for g in gathered:
        assert torch.equal(
            g, expected
        )  # bit-identical on every rank, own rows included
    partials = [_rows(rows, seed=10 + r) for r in range(4)]
    # Every rank sends its quarters first so the mailbox is complete before receives.
    for o, p in zip(owners, partials):
        q = wire.quantize_rows(p, codec=codec)
        for dst in range(4):
            if dst != o.rank:
                mailbox["p2p"][(o.rank, dst)] = q[
                    dst * rows // 4 : (dst + 1) * rows // 4
                ].clone()
    for o, p in zip(owners, partials):
        out = o.reduce_scatter(p)
        shards = torch.stack(
            [
                wire.quantize_rows(partials[src], codec=codec)[
                    o.rank * rows // 4 : (o.rank + 1) * rows // 4
                ]
                for src in range(4)
            ]
        )
        assert torch.equal(out, wire.accumulate_rows(shards, codec=codec))
        assert o.rs_count == 1 and o.ag_count == 1
        sends = [c for c in o.comm.calls if c[0] == "send"]
        assert [c[1] for c in sends] == [d for d in range(4) if d != o.rank]


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("codec", wire.CODECS)
def test_triton_kernels_match_reference_on_gpu(codec):
    x = _rows(64, seed=3).to("cuda")
    packed = wire.quantize_rows(x, codec=codec)
    reference = torch.empty_like(packed)
    wire._quantize_rows_reference(x, reference, codec=codec)
    assert torch.equal(packed, reference)
    back = wire.dequantize_rows(packed, codec=codec)
    assert torch.equal(back, wire._dequantize_rows_reference(packed, codec=codec))
    shards = torch.stack(
        [packed, packed.roll(1, 0), packed.roll(2, 0), packed.roll(3, 0)]
    )
    out = wire.accumulate_rows(shards, codec=codec)
    expected = sum(
        wire._dequantize_rows_reference(shards[s], dtype=torch.float32, codec=codec)
        for s in range(4)
    ).to(torch.bfloat16)
    assert torch.equal(out, expected)
    torch.accelerator.synchronize()
