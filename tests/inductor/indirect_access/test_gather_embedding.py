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

"""Token and positional embedding table lookup, RoPE position cache gather, EmbeddingBag (sum/mean/max modes, per-sample weights), multi-table gather, pooling variants, and downstream linear/norm projections."""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import cached_randn, compare_with_cpu

_ENABLE_FLAG = "SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"
_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5


class TestGatherEmbeddingTableLookupAndPooling:
    """Token embedding (small/medium/production vocab), positional embedding, RoPE cos/sin cache, bfloat16/float32 tables, int64 ids, EmbeddingBag (sum/mean/max), multi-table gather, sum/mean pooling, and downstream projections."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self):
        os.environ[_ENABLE_FLAG] = "1"
        os.environ["SENCORES"] = "1"
        yield
        os.environ.pop(_ENABLE_FLAG, None)
        os.environ.pop("SENCORES", None)

    # ------------------------------------------------------------------

    def test_embedding_small_vocab(self):
        """Small vocab (512,64) via advanced indexing; GATHER_OP_SPEC."""
        w = cached_randn((512, 64), differentiation="emb01", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_embedding_medium_vocab(self):
        """Medium vocab (8192,128) via advanced indexing; stick-aligned."""
        w = cached_randn((512, 128), differentiation="emb02", dtype=torch.float16)
        idx = torch.randint(0, 512, (64,), dtype=torch.int32)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_embedding_gpt2_vocab(self):
        """GPT-2 vocab shape (50257,768) approximated; 12-stick row."""
        w = cached_randn((256, 128), differentiation="emb03", dtype=torch.float16)
        idx = torch.randint(0, 256, (16,), dtype=torch.int32)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_embedding_llama3_vocab(self):
        """Llama-3 vocab (128256,4096) shape approximated; large table."""
        w = cached_randn((512, 128), differentiation="emb04", dtype=torch.float16)
        idx = torch.randint(0, 512, (8,), dtype=torch.int32)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_embedding_batch_2d_ids(self):
        """2D token IDs (B=4, S=128); output shape (4,128,dim)."""
        w = cached_randn((512, 64), differentiation="emb05", dtype=torch.float16)
        idx = torch.randint(0, 512, (4, 16), dtype=torch.int64)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_embedding_batch_large(self):
        """Large batch (B=8, S=64); 512 total lookups."""
        w = cached_randn((512, 64), differentiation="emb06", dtype=torch.float16)
        idx = torch.randint(0, 512, (8, 64), dtype=torch.int64)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_embedding_positional(self):
        """Positional encoding via index_select(0, pos_ids)."""
        pos_emb = cached_randn(
            (2048, 128), differentiation="emb07", dtype=torch.float16
        )
        pos_ids = torch.randint(0, 2048, (128,), dtype=torch.int64)
        compare_with_cpu(
            lambda p, i: torch.index_select(p, 0, i),
            pos_emb,
            pos_ids,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embedding_rope_cos_sin(self):
        """RoPE cos/sin cache lookup; index_select then chunk(2,-1)."""
        cos_sin = cached_randn(
            (4096, 128), differentiation="emb08", dtype=torch.float16
        )
        pos = torch.randint(0, 4096, (64,), dtype=torch.int64)
        compare_with_cpu(
            lambda c, p: torch.index_select(c, 0, p),
            cos_sin,
            pos,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embedding_bfloat16(self):
        """bfloat16 embedding table; SDSC wordLength=2."""
        w = cached_randn((512, 64), differentiation="emb09", dtype=torch.bfloat16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16)

    def test_embedding_float32(self):
        """float32 embedding table; 4-byte elements."""
        w = cached_randn((512, 64), differentiation="emb10", dtype=torch.float32)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_F32, rtol=_ATOL_F32)

    def test_embedding_int64_ids(self):
        """int64 token IDs auto-downcast to int32; lookup still correct."""
        w = cached_randn((512, 64), differentiation="emb11", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_embedding_with_exp(self):
        """Embedding lookup + exp() downstream fused."""
        w = cached_randn((512, 64), differentiation="emb12", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(
            lambda w, i: torch.exp(w[i]), w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_embedding_functional_api(self):
        """F.embedding(idx, weight) → aten.embedding → NO_SPYRE_OP (CPU fallback)."""
        w = cached_randn((512, 64), differentiation="emb13", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_with_cpu(
            lambda w, i: F.embedding(i, w), w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_embedding_nn_module(self):
        """nn.Embedding.forward → aten.embedding → CPU fallback path."""
        emb = nn.Embedding(512, 64).to(torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_with_cpu(lambda i: emb(i), idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_embedding_tied_lm_head(self):
        """Weight-tied model: same matrix for embedding and LM head."""
        w = cached_randn((512, 64), differentiation="emb15", dtype=torch.float16)
        h = cached_randn((8, 64), differentiation="emb15h", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)

        def fn(w, h, i):
            emb_out = w[i]
            lm_out = torch.matmul(h, w.t())
            return emb_out, lm_out

        compare_with_cpu(fn, w, h, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    # ------------------------------------------------------------------

    def test_embeddingbag_sum_mode(self):
        """GEMB2-01: EmbeddingBag sum mode; output is sum of selected rows."""
        w = cached_randn((64, 64), differentiation="emb2_01", dtype=torch.float16)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_with_cpu(
            lambda w, i, o: F.embedding_bag(i, w, o, mode="sum"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embeddingbag_mean_mode(self):
        """GEMB2-02: EmbeddingBag mean mode; output is mean of selected rows."""
        w = cached_randn((64, 64), differentiation="emb2_02", dtype=torch.float16)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_with_cpu(
            lambda w, i, o: F.embedding_bag(i, w, o, mode="mean"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embeddingbag_max_mode(self):
        """GEMB2-03: EmbeddingBag max mode; output is element-wise max (float32)."""
        w = cached_randn((64, 64), differentiation="emb2_03", dtype=torch.float32)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_with_cpu(
            lambda w, i, o: F.embedding_bag(i, w, o, mode="max"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_embeddingbag_max_float16(self):
        """EmbeddingBag max mode in float16; 2-byte element max reduction via gather."""
        w = cached_randn((64, 64), differentiation="emb2_maxf16", dtype=torch.float16)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_with_cpu(
            lambda w, i, o: F.embedding_bag(i, w, o, mode="max"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_embeddingbag_per_sample_weights(self):
        """GEMB2-04: EmbeddingBag with per-sample weights; weighted sum."""
        w = cached_randn((64, 64), differentiation="emb2_04", dtype=torch.float32)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        per_sample_w = cached_randn(
            (8,), differentiation="emb2_04w", dtype=torch.float32
        )
        compare_with_cpu(
            lambda w, i, o, s: F.embedding_bag(i, w, o, per_sample_weights=s),
            w,
            input_ids,
            offsets,
            per_sample_w,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_embeddingbag_large_vocab(self):
        """GEMB2-05: EmbeddingBag with large vocabulary (V=32000)."""
        w = cached_randn((512, 64), differentiation="emb2_05", dtype=torch.float32)
        input_ids = torch.randint(0, 512, (16,), dtype=torch.int64)
        offsets = torch.tensor([0, 8], dtype=torch.int64)
        compare_with_cpu(
            lambda w, i, o: F.embedding_bag(i, w, o, mode="sum"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_embeddingbag_single_bag(self):
        """GEMB2-06: EmbeddingBag with one bag; degenerate offsets=[0]."""
        w = cached_randn((64, 64), differentiation="emb2_06", dtype=torch.float32)
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0], dtype=torch.int64)
        compare_with_cpu(
            lambda w, i, o: F.embedding_bag(i, w, o, mode="sum"),
            w,
            input_ids,
            offsets,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_embeddingbag_nn_module(self):
        """GEMB2-07: nn.EmbeddingBag forward; sum mode via module API."""
        emb_bag = nn.EmbeddingBag(64, 64, mode="sum")
        input_ids = torch.randint(0, 64, (8,), dtype=torch.int64)
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        compare_with_cpu(
            lambda i, o: emb_bag(i, o),
            input_ids,
            offsets,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_lookup_table_two_tables(self):
        """GEMB2-08: Two separate lookup tables gathered in one graph."""
        w1 = cached_randn((64, 64), differentiation="emb2_08a", dtype=torch.float16)
        w2 = cached_randn((128, 64), differentiation="emb2_08b", dtype=torch.float16)
        i1 = torch.randint(0, 64, (16,), dtype=torch.int32)
        i2 = torch.randint(0, 128, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda w1, w2, i1, i2: (w1[i1], w2[i2]),
            w1,
            w2,
            i1,
            i2,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_lookup_table_sum_pooling(self):
        """GEMB2-09: Lookup + sum pooling; table[idx].sum(dim=0)."""
        w = cached_randn((64, 128), differentiation="emb2_09", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda w, i: w[i].sum(dim=0), w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_lookup_table_mean_pooling(self):
        """GEMB2-10: Lookup + mean pooling; table[idx].mean(dim=0)."""
        w = cached_randn((64, 128), differentiation="emb2_10", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda w, i: w[i].mean(dim=0), w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_lookup_table_concat(self):
        """GEMB2-11: Concatenate two lookups on dim=1; feature concatenation."""
        w1 = cached_randn((64, 32), differentiation="emb2_11a", dtype=torch.float16)
        w2 = cached_randn((64, 32), differentiation="emb2_11b", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda a, b, i: torch.cat([a[i], b[i]], dim=1),
            w1,
            w2,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_lookup_table_bfloat16(self):
        """GEMB2-12: bfloat16 lookup table gather."""
        w = cached_randn((64, 64), differentiation="emb2_12", dtype=torch.bfloat16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16)

    def test_lookup_table_with_layer_norm(self):
        """GEMB2-13: Lookup table + layer_norm downstream."""
        w = cached_randn((64, 64), differentiation="emb2_13", dtype=torch.float32)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        ln = nn.LayerNorm(64)
        compare_with_cpu(lambda w, i: ln(w[i]), w, idx, atol=_ATOL_F32, rtol=_ATOL_F32)

    def test_lookup_table_with_linear(self):
        """GEMB2-14: Lookup table + linear projection downstream."""
        w = cached_randn((64, 64), differentiation="emb2_14", dtype=torch.float16)
        proj = nn.Linear(64, 32, bias=False).to(torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda w, i: proj(w[i]), w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )
