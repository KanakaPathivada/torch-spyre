# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Production inference patterns: MoE expert routing (random and topk-based), paged KV cache (25 pool shapes, 2D slot index, block table), vLLM serving (15 scenarios), FMS 4D/5D layouts, multi-layer transformer KV gather, and real model shapes (Granite/Llama/Mistral)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import DEVICE, cached_randn, compare_with_cpu

_ENABLE_FLAG = "SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"
_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5


class TestGatherProductionInferencePatterns:
    """MoE routing (random ids and topk-based), paged KV cache (diverse pool shapes, 2D slot index, block table), vLLM serving scenarios, FMS 4D/5D layouts, multi-layer transformer KV gather, and real model shapes (Granite/Llama/Mistral)."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self):
        os.environ[_ENABLE_FLAG] = "1"
        os.environ["SENCORES"] = "1"
        yield
        os.environ.pop(_ENABLE_FLAG, None)
        os.environ.pop("SENCORES", None)
        os.environ.pop("LX_PLANNING", None)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "shape,P,sencores,dtype,atol,diff_key",
        [
            ((4, 64, 32), 8, "1", torch.float16, _ATOL_F16, "moe01"),
            ((64, 64, 64), 32, "1", torch.float16, _ATOL_F16, "moe02"),
            ((256, 64, 32), 16, "1", torch.float16, _ATOL_F16, "moe03"),
            ((8, 64, 32), 32, "1", torch.float16, _ATOL_F16, "moe04"),
            ((8, 64, 64), 32, "1", torch.bfloat16, _ATOL_BF16, "moe05"),
            ((8, 64, 64), 32, "4", torch.float16, _ATOL_F16, "moe06"),
            ((8, 64, 128), 32, "1", torch.float16, _ATOL_F16, "moe07"),
            ((256, 64, 32), 16, "1", torch.float16, _ATOL_F16, "moe08"),
        ],
    )
    def test_moe_expert_routing(self, shape, P, sencores, dtype, atol, diff_key):
        """MoE expert weight gather: expert_w[ids] across shapes, dtypes, and core counts."""
        os.environ["SENCORES"] = sencores
        w = cached_randn(shape, differentiation=diff_key, dtype=dtype)
        ids = torch.randint(0, shape[0], (P,), dtype=torch.int32)
        compare_with_cpu(lambda w, i: w[i], w, ids, atol=atol, rtol=atol)

    def test_moe_expert_aggregation(self):
        """Route via gather then weighted-sum aggregation."""
        w = cached_randn((8, 64, 32), differentiation="moe09", dtype=torch.float16)
        ids = torch.randint(0, 8, (16,), dtype=torch.int32)
        scores = torch.randn(16, dtype=torch.float16).softmax(dim=0)
        compare_with_cpu(
            lambda w, i, s: (w[i] * s[:, None, None]).sum(dim=0),
            w,
            ids,
            scores,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_moe_multi_layer(self):
        """Two consecutive MoE routing layers; two GATHER_OP_SPECs."""
        w1 = cached_randn((8, 64, 32), differentiation="moe10a", dtype=torch.float16)
        w2 = cached_randn((8, 32, 16), differentiation="moe10b", dtype=torch.float16)
        ids = torch.randint(0, 8, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda w1, w2, i: (w1[i], w2[i]),
            w1,
            w2,
            ids,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "kv_shape,P,dtype,atol,sencores,diff_key",
        [
            ((256, 8, 64), 128, torch.float16, _ATOL_F16, "1", "gpa01"),
            ((1024, 8, 64), 128, torch.float16, _ATOL_F16, "1", "gpa02"),
            ((512, 16, 128), 256, torch.float16, _ATOL_F16, "1", "gpa03"),
            ((512, 8, 64), 12, torch.float16, _ATOL_F16, "1", "gpa04"),
            ((512, 8, 64), 256, torch.float16, _ATOL_F16, "1", "gpa05"),
            ((1024, 8, 64), 256, torch.float16, _ATOL_F16, "1", "gpa07"),
            ((1024, 1, 64), 256, torch.float16, _ATOL_F16, "1", "gpa08"),
            ((512, 16, 128), 256, torch.float16, _ATOL_F16, "1", "gpa09"),
            ((512, 8, 256), 128, torch.float16, _ATOL_F16, "1", "gpa10"),
            ((512, 8, 64), 128, torch.bfloat16, _ATOL_BF16, "1", "gpa11"),
            ((512, 8, 64), 128, torch.float16, _ATOL_F16, "1", "gpa17"),
            ((512, 8, 64), 128, torch.float16, _ATOL_F16, "4", "gpa18"),
            ((512, 8, 64), 256, torch.float16, _ATOL_F16, "32", "gpa19"),
            ((128, 8, 16, 64), 64, torch.float16, _ATOL_F16, "1", "gpa23"),
        ],
    )
    def test_paged_kv_pool_shape(self, kv_shape, P, dtype, atol, sencores, diff_key):
        """Paged KV pool gather: kv[idx] across pool shapes, head configs, dtypes, cores."""
        os.environ["SENCORES"] = sencores
        kv = cached_randn(kv_shape, differentiation=diff_key, dtype=dtype)
        idx = torch.randint(0, kv_shape[0], (P,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=atol, rtol=atol)

    def test_paged_kv_chunked_prefill(self):
        """Chunked prefill: chunk_size=64; 4 separate gather calls."""
        kv = cached_randn((512, 8, 64), differentiation="gpa06", dtype=torch.float16)

        def fn(x, i):
            return x[i]

        for chunk_start in range(0, 256, 64):
            idx = torch.randint(0, 512, (64,), dtype=torch.int32)
            compare_with_cpu(fn, kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_paged_kv_block_table(self):
        """Block table: slot = block_table[b, t] * page_size + offset."""
        kv = cached_randn((512, 8, 64), differentiation="gpa12", dtype=torch.float16)
        block_table = torch.randint(0, 32, (4, 8), dtype=torch.int32)
        page_size = 16
        flat_slots = (block_table * page_size).flatten() % 512
        compare_with_cpu(
            lambda x, i: x[i], kv, flat_slots, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_paged_kv_page_reuse(self):
        """Prefix caching: same page referenced by multiple requests."""
        kv = cached_randn((512, 8, 64), differentiation="gpa13", dtype=torch.float16)
        shared_page = 42
        idx = torch.full((32,), shared_page, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=0, rtol=0)

    def test_paged_kv_online_softmax(self):
        """K gather + attention scores + softmax."""
        kv = cached_randn((512, 8, 64), differentiation="gpa14", dtype=torch.float16)
        q = cached_randn((4, 8, 64), differentiation="gpa14q", dtype=torch.float16)
        idx = torch.randint(0, 512, (64,), dtype=torch.int32)
        compare_with_cpu(
            lambda kv, q, i: torch.softmax(
                (q @ kv[i].transpose(-1, -2)).float(), dim=-1
            ).half(),
            kv,
            q,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_paged_kv_full_attention(self):
        """Full paged attention: gather K + V + attention + weighted V sum."""
        kv_k = cached_randn((512, 8, 64), differentiation="gpa15k", dtype=torch.float16)
        kv_v = cached_randn((512, 8, 64), differentiation="gpa15v", dtype=torch.float16)
        q = cached_randn((4, 8, 64), differentiation="gpa15q", dtype=torch.float16)
        idx = torch.randint(0, 512, (64,), dtype=torch.int32)
        compare_with_cpu(
            lambda k, v, q, i: (
                torch.softmax((q @ k[i].transpose(-1, -2)).float() / 8.0, dim=-1).half()
                @ v[i]
            ),
            kv_k,
            kv_v,
            q,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_paged_kv_named_dims(self):
        """Named dims on KV and slot_idxs."""
        import torch_spyre._inductor.propagate_named_dims as _pnd

        kv = cached_randn((512, 8, 64), differentiation="gpa16", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 64,), dtype=torch.int32)
        _pnd.name_tensor_dims(kv, {"cache": 512, "H": 8, "D": 64})
        _pnd.name_tensor_dims(idx, {"slots": 4 * 64})
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_paged_kv_lx_planning(self):
        """LX_PLANNING=1; slot_idxs must stay in HBM not LX."""
        os.environ["LX_PLANNING"] = "1"
        kv = cached_randn((512, 8, 64), differentiation="gpa20", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_paged_kv_two_gathers(self):
        """K and V gathered from same pool; two GATHER_OP_SPECs."""
        k = cached_randn((512, 8, 64), differentiation="gpa21k", dtype=torch.float16)
        v = cached_randn((512, 8, 64), differentiation="gpa21v", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 32,), dtype=torch.int32)
        compare_with_cpu(
            lambda k, v, i: (k[i], v[i]),
            k,
            v,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_paged_kv_dynamic_seqlen(self):
        """Dynamic Lk; compile once, run at Lk=32 and Lk=64."""
        kv = cached_randn((512, 8, 64), differentiation="gpa22", dtype=torch.float16)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        for lk in (32, 64):
            idx = torch.randint(0, 512, (lk,), dtype=torch.int32)
            expected = kv[idx]
            result = fn(kv.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, expected, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_paged_kv_5d_split_kv(self):
        """FMS 5D combined K+V along dim=1; gather + split downstream."""
        kv = cached_randn(
            (128, 2, 8, 16, 64), differentiation="gpa24", dtype=torch.float16
        )
        idx = torch.randint(0, 128, (4 * 16,), dtype=torch.int32)
        compare_with_cpu(
            lambda x, i: torch.unbind(x[i], dim=1),
            kv,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_paged_kv_sentinel_boundary(self):
        """Page boundary values; no off-by-one at slot boundaries."""
        kv = cached_randn((512, 8, 64), differentiation="gpa25", dtype=torch.float16)
        page_size = 16
        boundary_slots = torch.tensor(
            [page_size - 1, page_size, 2 * page_size - 1, 2 * page_size],
            dtype=torch.int32,
        )
        compare_with_cpu(lambda x, i: x[i], kv, boundary_slots, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_vllm_token_embedding(self):
        """On-device token embedding lookup; GATHER_OP_SPEC."""
        vocab = cached_randn((512, 128), differentiation="vllm01", dtype=torch.float16)
        token_ids = torch.randint(0, 512, (128,), dtype=torch.int64)
        compare_with_cpu(
            lambda v, i: v[i], vocab, token_ids, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_vllm_kv_cache_write(self):
        """KV cache write via index_put; written positions match."""
        cache = torch.zeros(1, 8, 128, 64, dtype=torch.float16)
        key = torch.randn(1, 8, 11, 64, dtype=torch.float16)
        pos = torch.arange(11, dtype=torch.int64)

        def fn(cache, key, pos):
            return cache.index_copy(2, pos, key[:, :, :11, :])

        compare_with_cpu(fn, cache, key, pos, atol=0, rtol=0)

    def test_vllm_kv_prefill_read(self):
        """Prefill KV read (sequential contiguous slots)."""
        kv = cached_randn((512, 8, 64), differentiation="vllm03", dtype=torch.float16)
        idx = torch.arange(256, dtype=torch.int32) % 512
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_vllm_kv_decode_read(self):
        """Decode via paged slot indices; B=12, Lk=1."""
        kv = cached_randn((512, 8, 64), differentiation="vllm04", dtype=torch.float16)
        idx = torch.randint(0, 512, (12,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_vllm_logit_extraction(self):
        """Decode-step last-token logit extraction."""
        logits = cached_randn(
            (12, 4, 512), differentiation="vllm05", dtype=torch.float16
        )
        compare_with_cpu(lambda x: x[:, -1, :], logits, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_vllm_batch_variable(self):
        """Ragged-batch: per-request variable active lengths."""
        kv = cached_randn((512, 8, 64), differentiation="vllm06", dtype=torch.float16)
        idx = torch.randint(0, 512, (48,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_vllm_4d_kv_layout(self):
        """vLLM native 4D KV layout (num_blocks, block_sz, kv_h, head_d)."""
        kv = cached_randn(
            (64, 16, 8, 64), differentiation="vllm07", dtype=torch.float16
        )
        idx = torch.randint(0, 64, (4 * 16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_vllm_prefill_decode_combined(self):
        """Combined prefill+decode batch in one gather call."""
        kv = cached_randn((512, 8, 64), differentiation="vllm08", dtype=torch.float16)
        idx = torch.randint(0, 512, (128 + 4,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_vllm_rope_position_lookup(self):
        """RoPE position cache; index_select → GATHER_OP_SPEC."""
        cos_sin = cached_randn(
            (4096, 128), differentiation="vllm09", dtype=torch.float16
        )
        pos = torch.randint(0, 4096, (128,), dtype=torch.int64)
        compare_with_cpu(
            lambda c, p: torch.index_select(c, 0, p),
            cos_sin,
            pos,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_vllm_beam_search_reorder(self):
        """Beam search KV reordering via index_select at batch dim."""
        past_kv = cached_randn(
            (4, 32, 8, 64), differentiation="vllm10", dtype=torch.float16
        )
        beam_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        compare_with_cpu(
            lambda x, i: torch.index_select(x, 0, i),
            past_kv,
            beam_idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_vllm_llama31_8b(self):
        """Llama-3.1-8B shapes; token emb + KV cache dual gather."""
        kv = cached_randn(
            (512, 8, 128), differentiation="vllm11kv", dtype=torch.float16
        )
        emb = cached_randn((512, 128), differentiation="vllm11emb", dtype=torch.float16)
        slot_idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        tok_idx = torch.randint(0, 512, (64,), dtype=torch.int64)
        compare_with_cpu(
            lambda kv, emb, si, ti: (kv[si], emb[ti]),
            kv,
            emb,
            slot_idx,
            tok_idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_vllm_mistral_small_24b(self):
        """Mistral-Small-3.2-24B sliding-window KV gather."""
        kv = cached_randn((512, 16, 128), differentiation="vllm12", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 64,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_vllm_granite_8b(self):
        """Granite-3.3-8b GQA (kv_heads=8, q_heads=32)."""
        kv = cached_randn((512, 8, 128), differentiation="vllm13", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 64,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_vllm_continuous_batching(self):
        """Continuous batching: B requests, variable active KV lengths."""
        kv = cached_randn((512, 8, 64), differentiation="vllm14", dtype=torch.float16)
        idx = torch.randint(0, 512, (96,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_vllm_speculative_decode(self):
        """Speculative decode; gather target probs at draft positions."""
        probs = cached_randn((512, 64), differentiation="vllm15", dtype=torch.float16)
        draft_tokens = torch.randint(0, 64, (5,), dtype=torch.int64)
        compare_with_cpu(
            lambda p, i: torch.gather(
                p[:5], 1, i.unsqueeze(0).expand(5, -1)
            ).diagonal(),
            probs,
            draft_tokens,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def test_fms_5d_split_kv(self):
        """FMS 5D combined K+V (pages,2,kv_h,page_sz,D); split after."""
        kv = cached_randn(
            (128, 2, 8, 16, 128), differentiation="fms02", dtype=torch.float16
        )
        idx = torch.randint(0, 128, (4 * 16,), dtype=torch.int32)
        compare_with_cpu(
            lambda x, i: torch.unbind(x[i], dim=1),
            kv,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_fms_named_dims(self):
        """Named dims on FMS layout (pages, kv_h, page_sz, D)."""
        import torch_spyre._inductor.propagate_named_dims as _pnd

        kv = cached_randn(
            (128, 8, 16, 128), differentiation="fms07", dtype=torch.float16
        )
        idx = torch.randint(0, 128, (4 * 16,), dtype=torch.int32)
        _pnd.name_tensor_dims(kv, {"pages": 128, "kv_h": 8, "page_sz": 16, "D": 128})
        _pnd.name_tensor_dims(idx, {"slots": 4 * 16})
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    @pytest.mark.parametrize(
        "P,dtype,atol,sencores,diff_key",
        [
            (64, torch.float16, _ATOL_F16, "1", "fms01"),
            (48, torch.float16, _ATOL_F16, "1", "fms03"),
            (4 * 64, torch.float16, _ATOL_F16, "1", "fms04"),
            (1, torch.float16, _ATOL_F16, "1", "fms05"),
            (4 * 16, torch.bfloat16, _ATOL_BF16, "1", "fms06"),
            (4 * 64, torch.float16, _ATOL_F16, "32", "fms08"),
        ],
    )
    def test_fms_4d_layout(self, P, dtype, atol, sencores, diff_key):
        """FMS 4D KV (128,8,16,128) gather across dtypes, output sizes, and core counts."""
        os.environ["SENCORES"] = sencores
        kv = cached_randn((128, 8, 16, 128), differentiation=diff_key, dtype=dtype)
        idx = torch.randint(0, 128, (P,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=atol, rtol=atol)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "kv_shape,P,diff_key",
        [
            ((512, 8, 64), 11, "mdl01"),
            ((512, 8, 64), 1, "mdl02"),
            ((512, 8, 128), 41, "mdl03"),
            ((512, 8, 128), 14, "mdl04"),
            ((512, 16, 128), 128, "mdl05"),
            ((512, 8, 64), 41, "mdl09"),
        ],
    )
    def test_model_kv_prefill_decode(self, kv_shape, P, diff_key):
        """Model-specific KV gather at production prefill/decode sizes."""
        kv = cached_randn(kv_shape, differentiation=diff_key, dtype=torch.float16)
        idx = torch.randint(0, kv_shape[0], (P,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_masked_scatter_mistral_multimodal(self):
        """Mistral-Small multimodal: gather image token embeddings at mask positions."""
        image_features = cached_randn(
            (16, 128), differentiation="mdl06", dtype=torch.float16
        )
        idx = torch.arange(16, dtype=torch.int32)
        compare_with_cpu(
            lambda x, i: x[i], image_features, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_rope_index_select_float32(self):
        """SpyreRotaryEmbedding float32: cos_sin_cache.index_select(0, pos); 4-byte elements."""
        cos_sin = cached_randn(
            (4096, 128), differentiation="mdl07", dtype=torch.float32
        )
        pos = torch.randint(0, 4096, (128,), dtype=torch.int64)
        compare_with_cpu(
            lambda c, p: torch.index_select(c, 0, p),
            cos_sin,
            pos,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_lm_head_logit_extraction(self):
        """Last-token logit extraction hidden[:, -1, :]."""
        hidden = cached_randn(
            (12, 4, 128), differentiation="mdl08", dtype=torch.float16
        )
        compare_with_cpu(lambda h: h[:, -1, :], hidden, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_kv_cache_decode_growing(self):
        """Growing KV cache: 3 separate decode-step gathers at positions 0, 10, 100."""
        kv = cached_randn((512, 8, 64), differentiation="mdl10", dtype=torch.float16)
        for slot in [0, 10, 100]:
            idx = torch.tensor([slot], dtype=torch.int32)
            compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_attention_qkv_split(self):
        """QKV split via tensor.split; no gather needed."""
        qkv = cached_randn((1, 4, 384), differentiation="mdl11", dtype=torch.float16)
        q_size, k_size, v_size = 128, 128, 128
        compare_with_cpu(
            lambda x: torch.split(x, [q_size, k_size, v_size], dim=-1),
            qkv,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_token_unpadding(self):
        """Packed token gather: each seq pos from flat packed buffer."""
        packed = cached_randn((64, 128), differentiation="mdl12", dtype=torch.float16)
        pos_ids = torch.randint(0, 64, (4, 16), dtype=torch.int64)
        compare_with_cpu(
            lambda p, i: p[i], packed, pos_ids, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_token_repadding(self):
        """Token repadding: larger packed buffer (128,256) gathered at (8,32) batch positions."""
        packed = cached_randn((128, 256), differentiation="mdl13", dtype=torch.float16)
        pos_ids = torch.randint(0, 128, (8, 32), dtype=torch.int64)
        compare_with_cpu(
            lambda p, i: p[i], packed, pos_ids, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    @pytest.mark.parametrize(
        "kv_shape,idx_shape,diff_key",
        [
            ((512, 32, 128), (4, 16), "gpa27"),
            ((1024, 32, 128), (12, 64), "gpa28"),
            ((512, 32, 128), (4, 32), "gpa29"),
        ],
    )
    def test_paged_kv_2d_slot_index(self, kv_shape, idx_shape, diff_key):
        """Paged KV: 2D (B,Lk) slot_idxs into 3D cache — FRS paged gather pattern."""
        kv = cached_randn(kv_shape, differentiation=diff_key, dtype=torch.float16)
        slot_idxs = torch.randint(0, kv_shape[0], idx_shape, dtype=torch.int32)
        compare_with_cpu(
            lambda x, i: x[i], kv, slot_idxs, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_topk_moe_routing_top1(self):
        """MoE top-1 routing: softmax → topk(1) → index → gather expert weights."""
        n_experts = 8
        expert_w = cached_randn(
            (n_experts, 64, 32), differentiation="topk01", dtype=torch.float16
        )
        logits = cached_randn(
            (4, n_experts), differentiation="topk01l", dtype=torch.float16
        )

        def fn(expert_w, logits):
            ids = (
                torch.topk(logits.float(), 1, dim=-1)
                .indices.squeeze(-1)
                .to(torch.int32)
            )
            return expert_w[ids]

        compare_with_cpu(fn, expert_w, logits, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_topk_moe_routing_top2(self):
        """MoE top-2 routing: topk(2) flattened → gather 2 experts per token."""
        n_experts = 8
        expert_w = cached_randn(
            (n_experts, 32, 64), differentiation="topk02", dtype=torch.float16
        )
        logits = cached_randn(
            (4, n_experts), differentiation="topk02l", dtype=torch.float16
        )

        def fn(expert_w, logits):
            ids = (
                torch.topk(logits.float(), 2, dim=-1).indices.flatten().to(torch.int32)
            )
            return expert_w[ids]

        compare_with_cpu(fn, expert_w, logits, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_topk_moe_weighted_routing(self):
        """MoE top-2 weighted: topk scores × gathered expert outputs → weighted sum."""
        n_experts = 8
        expert_w = cached_randn(
            (n_experts, 64), differentiation="topk03", dtype=torch.float16
        )
        logits = cached_randn(
            (4, n_experts), differentiation="topk03l", dtype=torch.float16
        )

        def fn(expert_w, logits):
            topk_out = torch.topk(logits.softmax(dim=-1).float(), 2, dim=-1)
            ids = topk_out.indices.to(torch.int32)
            weights = topk_out.values.half()
            selected = expert_w[ids.flatten()].view(4, 2, 64)
            return (selected * weights.unsqueeze(-1)).sum(dim=1)

        compare_with_cpu(fn, expert_w, logits, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_topk_speculative_candidate_embed(self):
        """Speculative decode: topk draft candidates → gather their token embeddings."""
        embed = cached_randn((512, 128), differentiation="topk04", dtype=torch.float16)
        scores = cached_randn((512,), differentiation="topk04s", dtype=torch.float16)

        def fn(embed, scores):
            candidate_ids = torch.topk(scores.float(), 16).indices.to(torch.int32)
            return embed[candidate_ids]

        compare_with_cpu(fn, embed, scores, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_topk_beam_search_kv_reorder(self):
        """Beam search: topk beam scores → gather KV rows for top beams."""
        kv = cached_randn((8, 8, 64), differentiation="topk05", dtype=torch.float16)
        beam_scores = cached_randn((8,), differentiation="topk05s", dtype=torch.float16)

        def fn(kv, scores):
            beam_ids = torch.topk(scores.float(), 4).indices.to(torch.int32)
            return kv[beam_ids]

        compare_with_cpu(fn, kv, beam_scores, atol=_ATOL_F16, rtol=_ATOL_F16)

    # ------------------------------------------------------------------

    def test_multi_layer_shared_slot_index(self):
        """4 transformer layers; same slot_idxs reused with separate KV cache per layer."""
        pool, H, D = 256, 8, 64
        slot_idxs = torch.randint(0, pool, (32,), dtype=torch.int32)
        for layer_id in range(4):
            k = cached_randn(
                (pool, H, D), differentiation=f"mlt01k{layer_id}", dtype=torch.float16
            )
            v = cached_randn(
                (pool, H, D), differentiation=f"mlt01v{layer_id}", dtype=torch.float16
            )
            compare_with_cpu(
                lambda k, v, s: (k[s], v[s]),
                k,
                v,
                slot_idxs,
                atol=_ATOL_F16,
                rtol=_ATOL_F16,
            )

    def test_multi_layer_growing_decode_history(self):
        """Decode grows: same KV cache read at 4 increasing history lengths across layers."""
        pool, H, D = 512, 8, 64
        kv = cached_randn((pool, H, D), differentiation="mlt02", dtype=torch.float16)
        for step in (1, 8, 32, 64):
            slot_idxs = torch.randint(0, pool, (step,), dtype=torch.int32)
            compare_with_cpu(
                lambda x, i: x[i], kv, slot_idxs, atol=_ATOL_F16, rtol=_ATOL_F16
            )

    def test_multi_layer_4d_kv_per_layer(self):
        """4-layer 4D KV (pool, H, blk, D); per-layer gather at decode with unique tensors."""
        pool, H, blk, D = 64, 8, 4, 32
        for layer_id in range(4):
            kv = cached_randn(
                (pool, H, blk, D),
                differentiation=f"mlt03_{layer_id}",
                dtype=torch.float16,
            )
            idx = torch.randint(0, pool, (8,), dtype=torch.int32)
            compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_multi_layer_bfloat16_kv(self):
        """4 transformer layers; bfloat16 KV caches, same slot pattern per layer."""
        pool, H, D = 256, 8, 64
        slot_idxs = torch.randint(0, pool, (16,), dtype=torch.int32)
        for layer_id in range(4):
            kv = cached_randn(
                (pool, H, D), differentiation=f"mlt04_{layer_id}", dtype=torch.bfloat16
            )
            compare_with_cpu(
                lambda x, i: x[i], kv, slot_idxs, atol=_ATOL_BF16, rtol=_ATOL_BF16
            )

    def test_multi_layer_parallel_k_v_per_layer(self):
        """2 transformer layers; K and V gathered in parallel within each layer call."""
        pool, H, D = 256, 8, 64
        slot_idxs = torch.randint(0, pool, (32,), dtype=torch.int32)
        for layer_id in range(2):
            k = cached_randn(
                (pool, H, D), differentiation=f"mlt05k{layer_id}", dtype=torch.float16
            )
            v = cached_randn(
                (pool, H, D), differentiation=f"mlt05v{layer_id}", dtype=torch.float16
            )
            compare_with_cpu(
                lambda k, v, s: (k[s], v[s]),
                k,
                v,
                slot_idxs,
                atol=_ATOL_F16,
                rtol=_ATOL_F16,
            )

    def test_float32_kv_cache_int32_index(self):
        """float32 3D KV pool (512,8,64) with int32 index; 4-byte element gather."""
        pool, H, D = 512, 8, 64
        kv = cached_randn((pool, H, D), differentiation="f32kv01", dtype=torch.float32)
        slots = torch.randint(0, pool, (32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, slots, atol=_ATOL_F32, rtol=_ATOL_F32)
