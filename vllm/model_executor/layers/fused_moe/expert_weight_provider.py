# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ExpertWeightResult:
    """GPU-resident expert weights ready for kernel consumption."""

    w1: torch.Tensor
    w2: torch.Tensor
    topk_ids: torch.Tensor
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None


class CachedWeightProvider:
    """GPU LRU cache backed by CPU pinned memory.

    Keeps capacity expert weight tensors in a fixed-size GPU scratch
    buffer. All expert weights reside in CPU pinned memory; only the N
    hottest experts are mirrored into the GPU buffer.

    Uses LFRU (frequency-weighted LRU) eviction: score = freq / age.
    This prevents early layers from monopolizing the cache — a known
    problem with pure LRU in sequential MoE execution where early
    layers always appear "recently used."

    On each forward pass, prepare() identifies which experts are needed,
    copies any misses from CPU to GPU (evicting the lowest-scored entry
    when the buffer is full), and returns an ExpertWeightResult with
    remapped topk_ids whose values are GPU-buffer slot indices.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        num_experts = w13_weight.size(0)
        if capacity <= 0:
            raise ValueError("CachedWeightProvider capacity must be positive.")
        if w2_weight.size(0) != num_experts:
            raise ValueError("w13_weight and w2_weight must have matching experts.")
        if (w13_scale is None) != (w2_scale is None):
            raise ValueError("w13_scale and w2_scale must be provided together.")

        self.capacity = capacity
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0

        if w13_weight.device.type == "cpu":
            cuda_device = torch.accelerator.current_accelerator()
        else:
            cuda_device = w13_weight.device
        self._cpu_w13 = self._to_pinned_cpu(w13_weight)
        self._cpu_w2 = self._to_pinned_cpu(w2_weight)

        self._buf_w13: torch.Tensor = torch.empty(
            capacity,
            *w13_weight.shape[1:],
            dtype=w13_weight.dtype,
            device=cuda_device,
        )
        self._buf_w2: torch.Tensor = torch.empty(
            capacity,
            *w2_weight.shape[1:],
            dtype=w2_weight.dtype,
            device=cuda_device,
        )

        if w13_scale is not None and w2_scale is not None:
            self._validate_expert_tensor(w13_scale, num_experts, "w13_scale")
            self._validate_expert_tensor(w2_scale, num_experts, "w2_scale")
            self._cpu_w13_scale: torch.Tensor | None = self._to_pinned_cpu(w13_scale)
            self._cpu_w2_scale: torch.Tensor | None = self._to_pinned_cpu(w2_scale)
            self._buf_w13_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w13_scale.shape[1:],
                dtype=w13_scale.dtype,
                device=cuda_device,
            )
            self._buf_w2_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w2_scale.shape[1:],
                dtype=w2_scale.dtype,
                device=cuda_device,
            )
        else:
            self._cpu_w13_scale = None
            self._cpu_w2_scale = None
            self._buf_w13_scale = None
            self._buf_w2_scale = None

        self._cpu_w13_bias, self._buf_w13_bias = self._make_aux_buffers(
            w13_bias,
            num_experts,
            capacity,
            cuda_device,
            "w13_bias",
        )
        self._cpu_w2_bias, self._buf_w2_bias = self._make_aux_buffers(
            w2_bias,
            num_experts,
            capacity,
            cuda_device,
            "w2_bias",
        )

        # LFRU state: {expert_id: [slot, freq, last_access_clock]}
        # Eviction score = freq / (clock - last_access + 1). Lower = evict first.
        self._lru: dict[int, list[int]] = {}
        self._clock: int = 0
        self._free_slots: list[int] = list(range(capacity))

        # Persistent GPU mapping tensor: _mapping[expert_id] = slot.
        self._mapping: torch.Tensor = torch.full(
            (num_experts,),
            -1,
            dtype=torch.int32,
            device=cuda_device,
        )

    @staticmethod
    def _validate_expert_tensor(
        tensor: torch.Tensor,
        num_experts: int,
        name: str,
    ) -> None:
        if tensor.ndim == 0 or tensor.size(0) != num_experts:
            raise ValueError(
                f"{name} must have {num_experts} entries in its first dimension."
            )

    @staticmethod
    def _to_pinned_cpu(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == "cpu" and tensor.is_pinned():
            return tensor
        if tensor.layout != torch.strided:
            raise ValueError("Expert cache backing tensors must use strided layout.")
        pinned = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )
        pinned.copy_(tensor)
        return pinned

    @classmethod
    def _make_aux_buffers(
        cls,
        tensor: torch.Tensor | None,
        num_experts: int,
        capacity: int,
        device: torch.device,
        name: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if tensor is None:
            return None, None
        cls._validate_expert_tensor(tensor, num_experts, name)
        cpu_tensor = cls._to_pinned_cpu(tensor)
        gpu_buffer = torch.empty(
            capacity,
            *tensor.shape[1:],
            dtype=tensor.dtype,
            device=device,
        )
        return cpu_tensor, gpu_buffer

    @property
    def buf_w13(self) -> torch.Tensor:
        return self._buf_w13

    @property
    def buf_w2(self) -> torch.Tensor:
        return self._buf_w2

    @property
    def buf_w13_scale(self) -> torch.Tensor | None:
        return self._buf_w13_scale

    @property
    def buf_w2_scale(self) -> torch.Tensor | None:
        return self._buf_w2_scale

    @property
    def buf_w13_bias(self) -> torch.Tensor | None:
        return self._buf_w13_bias

    @property
    def buf_w2_bias(self) -> torch.Tensor | None:
        return self._buf_w2_bias

    def invalidate(self, expert_id: int) -> None:
        """Remove *expert_id* from the cache, returning its slot to the free
        list.  No-op if the expert is not currently cached."""
        if expert_id in self._lru:
            entry = self._lru.pop(expert_id)
            self._free_slots.append(entry[0])
            self._mapping[expert_id] = -1

    def all_cached(self, topk_ids: torch.Tensor) -> bool:
        """Check if all experts in topk_ids are already cached.

        This is groundwork for a future CUDA graph dispatcher that can replay
        only warm-cache batches. Graph replay is not wired up yet.
        """
        unique_ids = topk_ids.unique().tolist()
        return all(eid in self._lru for eid in unique_ids)

    @torch.compiler.disable
    def prepare(self, topk_ids: torch.Tensor) -> ExpertWeightResult:
        """Populate the GPU buffer and return slot-remapped expert IDs.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            ExpertWeightResult with remapped topk_ids and GPU buffer refs.

        Raises:
            RuntimeError: if unique experts in the batch exceed capacity.
        """
        unique_ids = topk_ids.unique().tolist()
        if any(
            expert_id < 0 or expert_id >= self._num_experts for expert_id in unique_ids
        ):
            raise IndexError(
                "CachedWeightProvider received an expert ID outside "
                f"[0, {self._num_experts})."
            )
        requested_ids = set(unique_ids)
        if len(unique_ids) > self.capacity:
            raise RuntimeError(
                f"CachedWeightProvider: {len(unique_ids)} unique experts "
                f"requested but --moe-expert-cache-size={self.capacity}. "
                f"Set --moe-expert-cache-size >= {len(unique_ids)}."
            )

        for expert_id in unique_ids:
            if expert_id in self._lru:
                # Cache hit: update frequency and recency
                self._clock += 1
                entry = self._lru[expert_id]
                entry[1] += 1  # freq
                entry[2] = self._clock  # last access
                self.hits += 1
            else:
                # Cache miss: need to load expert
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    # Evict entry with lowest freq/age score
                    best_key = None
                    best_score = float("inf")
                    for key, (_, freq, last) in self._lru.items():
                        if key in requested_ids:
                            continue
                        age = self._clock - last + 1
                        score = freq / age
                        if score < best_score:
                            best_score = score
                            best_key = key
                    assert best_key is not None
                    slot = self._lru.pop(best_key)[0]
                    self._mapping[best_key] = -1

                # Copy expert weights from CPU to GPU slot
                self._buf_w13[slot].copy_(self._cpu_w13[expert_id])
                self._buf_w2[slot].copy_(self._cpu_w2[expert_id])
                if self._buf_w13_scale is not None:
                    assert self._cpu_w13_scale is not None
                    assert self._cpu_w2_scale is not None
                    assert self._buf_w2_scale is not None
                    self._buf_w13_scale[slot].copy_(self._cpu_w13_scale[expert_id])
                    self._buf_w2_scale[slot].copy_(self._cpu_w2_scale[expert_id])
                if self._buf_w13_bias is not None:
                    assert self._cpu_w13_bias is not None
                    self._buf_w13_bias[slot].copy_(self._cpu_w13_bias[expert_id])
                if self._buf_w2_bias is not None:
                    assert self._cpu_w2_bias is not None
                    self._buf_w2_bias[slot].copy_(self._cpu_w2_bias[expert_id])

                self._clock += 1
                self._lru[expert_id] = [slot, 1, self._clock]
                self._mapping[expert_id] = slot
                self.misses += 1

        total = self.hits + self.misses
        if total > 0:
            logger.debug(
                "Expert cache: %d hits, %d misses (%.1f%% hit rate)",
                self.hits,
                self.misses,
                100.0 * self.hits / total,
            )

        remapped_ids = self._mapping[topk_ids.long()].to(dtype=topk_ids.dtype)

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            topk_ids=remapped_ids,
            w1_scale=self._buf_w13_scale,
            w2_scale=self._buf_w2_scale,
            w1_bias=self._buf_w13_bias,
            w2_bias=self._buf_w2_bias,
        )
