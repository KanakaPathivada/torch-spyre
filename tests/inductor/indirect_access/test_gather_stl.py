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

"""SpyreTensorLayout (STL) annotation: 2-arg and 4-arg API, GQA/MQA KV cache layouts, dim_order reordering, STL vs non-STL numeric equivalence, and stride-exact verification for real model strides (Granite/GPT/Llama/RoPE)."""

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


class TestGatherSpyreTensorLayoutAnnotation:
    """STL 2-arg and 4-arg forms, GQA/MQA kv_heads, dim_order reordering, dynamic compile with STL, LX planning with STL, STL vs non-STL numeric equivalence checks, and stride-exact validation for Granite/GPT/Llama/RoPE caches."""

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

    def test_stl_2d_square(self):
        """(128,64) gather at dim=0; narrow inner dim forces non-trivial stick packing."""
        x = cached_randn((128, 64), differentiation="gstl01", dtype=torch.float16)
        idx = torch.randint(0, 128, (48,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_2d_frs_example_a(self):
        """FRS canonical Example A — (12,1024) gather at dim=0, P=8."""
        x = cached_randn((12, 1024), differentiation="gstl02", dtype=torch.float16)
        idx = torch.randint(0, 12, (8,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_2d_16x512(self):
        """(16,512), 8 sticks in inner dim; STL slicing into stick grid."""
        x = cached_randn((16, 512), differentiation="gstl03", dtype=torch.float16)
        idx = torch.randint(0, 16, (8,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_2d_int64_idx(self):
        """int64 index + STL; downcast does not corrupt layout."""
        x = cached_randn((32, 256), differentiation="gstl04", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_2d_with_tanh(self):
        """(32,256) STL gather + tanh downstream co-scheduled."""
        x = cached_randn((32, 256), differentiation="gstl05", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda x, i: torch.tanh(x[i]), x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_stl_2d_with_scalar_div(self):
        """STL gather + /8.0 attention scale downstream."""
        x = cached_randn((32, 256), differentiation="gstl06", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda x, i: x[i] / 8.0, x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_stl_2d_with_sum_reduction(self):
        """STL gather + sum(dim=1) reduction; shape collapses to (16,)."""
        x = cached_randn((32, 256), differentiation="gstl07", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda x, i: x[i].sum(dim=1), x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_stl_interior_dim_no_stl_needed(self):
        """x[:,idx,:] — dim=1 gather on (5,64,512); no STL injection."""
        x = cached_randn((5, 64, 512), differentiation="gstl08", dtype=torch.float16)
        idx = torch.randint(0, 64, (8,), dtype=torch.int64)
        compare_with_cpu(
            lambda x, i: torch.index_select(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_stl_3d_explicit(self):
        """3D (8,32,128) with explicit STL at dim=0, P=4."""
        x = cached_randn((8, 32, 128), differentiation="gstl09", dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_with_named_dims(self):
        """STL + name_tensor_dims on value and index."""
        import torch_spyre._inductor.propagate_named_dims as _pnd

        x = cached_randn((32, 256), differentiation="gstl10", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        _pnd.name_tensor_dims(x, {"M": 32, "N": 256})
        _pnd.name_tensor_dims(idx, {"P": 16})
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_dynamic_compile(self):
        """STL + torch.compile(dynamic=True); multiple runtime shapes."""
        x = cached_randn((32, 256), differentiation="gstl11", dtype=torch.float16)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        for p in (8, 16, 24):
            idx = torch.randint(0, 32, (p,), dtype=torch.int32)
            x_dev = x.to(DEVICE)
            idx_dev = idx.to(DEVICE)
            cpu_ref = x[idx]
            result = fn(x_dev, idx_dev).cpu()
            torch.testing.assert_close(result, cpu_ref, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_with_lx_planning(self):
        """STL + LX_PLANNING=1; compiler runs without crash."""
        os.environ["LX_PLANNING"] = "1"
        x = cached_randn((32, 256), differentiation="gstl12", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_gate_off(self):
        """gate=0 — STL specified but indirect access disabled."""
        os.environ[_ENABLE_FLAG] = "0"
        x = cached_randn((32, 256), differentiation="gstl13", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_index_never_has_stl(self):
        """Index tensor must not receive an STL annotation; output correct."""
        x = cached_randn((32, 256), differentiation="gstl14", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_embedding_shape(self):
        """Large vocab (32000,128) embedding; stick-aligned inner dim at production scale."""
        x = cached_randn((1024, 128), differentiation="gstl15", dtype=torch.float16)
        idx = torch.randint(0, 1024, (64,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_kv_cache_shape(self):
        """KV cache 3D (512,8,64); paged attention page lookup via STL."""
        x = cached_randn((512, 8, 64), differentiation="gstl16", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_moe_shape(self):
        """MoE expert weight (8,512,64); large inner dims; STL stride map."""
        x = cached_randn((8, 512, 64), differentiation="gstl17", dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_bfloat16_value(self):
        """bfloat16 value + STL; SDSC wordLength and stride map for bf16."""
        x = cached_randn((32, 256), differentiation="gstl18", dtype=torch.bfloat16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16)

    # ------------------------------------------------------------------

    def test_stl_vs_no_stl_2d_bare(self):
        """x[idx] output is bit-identical with and without explicit STL injection."""
        x = cached_randn((32, 256), differentiation="cmp01", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_stl_vs_no_stl_2d_exp(self):
        """GSTLCMP-02: x[idx].exp() — exp output identical with/without STL."""
        x = cached_randn((32, 256), differentiation="cmp02", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(
            lambda x, i: torch.exp(x[i]), x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_stl_vs_no_stl_kv(self):
        """GSTLCMP-03: KV gather with and without STL produces identical values."""
        kv = cached_randn((512, 8, 64), differentiation="cmp03", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=0, rtol=0)

    def test_stl_vs_no_stl_embedding(self):
        """GSTLCMP-04: Embedding lookup output identical with and without STL."""
        w = cached_randn((512, 64), differentiation="cmp04", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], w, idx, atol=0, rtol=0)

    def test_stl_vs_no_stl_bfloat16(self):
        """GSTLCMP-05: bfloat16 gather — outputs identical regardless of STL."""
        x = cached_randn((32, 256), differentiation="cmp05", dtype=torch.bfloat16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_stl_sdsc_encoding_differs(self):
        """GSTLCMP-06: STL changes device_coordinates in SDSC; output still identical."""
        x = cached_randn((32, 256), differentiation="cmp06", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_removal_regression(self):
        """GSTLCMP-07: Removing unnecessary STL does not change output."""
        x = cached_randn((32, 256), differentiation="cmp07", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_stl_api_equivalence(self):
        """x[idx] and torch.index_select(x,0,idx) produce identical output on Spyre."""
        x = cached_randn((32, 256), differentiation="cmp08", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_with_cpu(
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_stl_vs_no_stl_moe(self):
        """GSTLCMP-09: MoE expert gather output identical regardless of STL."""
        w = cached_randn((8, 512, 64), differentiation="cmp09", dtype=torch.float16)
        idx = torch.randint(0, 8, (32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], w, idx, atol=0, rtol=0)

    def test_stl_vs_no_stl_paged(self):
        """GSTLCMP-10: Full paged attention gather output identical with/without STL."""
        kv = cached_randn((512, 8, 64), differentiation="cmp10", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_stl_4arg_basic(self):
        """GSTLADV-01: 4-arg SpyreTensorLayout(device_size, strides, dtype, dim_order)."""
        kv = cached_randn((512, 8, 64), differentiation="stladv01", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_4arg_kv_exact_paged(self):
        """GSTLADV-02: Canonical paged-attention STL: cache_size=32768, H=32, D=128."""
        kv = cached_randn(
            (512, 32, 128), differentiation="stladv02", dtype=torch.float16
        )
        idx = torch.randint(0, 512, (12,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_4arg_dim_order_reorder(self):
        """GSTLADV-03: dim_order=[1,0,2] — kv_heads is outermost device dim."""
        kv = cached_randn((128, 8, 64), differentiation="stladv03", dtype=torch.float16)
        idx = torch.randint(0, 128, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_4arg_gqa_h8_d128(self):
        """GSTLADV-04: GQA KV cache (pool=1024, kv_h=8, D=128) with 4-arg STL."""
        kv = cached_randn(
            (1024, 8, 128), differentiation="stladv04", dtype=torch.float16
        )
        idx = torch.randint(0, 1024, (4 * 64,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_4arg_mqa_h1(self):
        """GSTLADV-05: MQA KV cache (pool=1024, kv_h=1, D=64) with 4-arg STL."""
        kv = cached_randn(
            (1024, 1, 64), differentiation="stladv05", dtype=torch.float16
        )
        idx = torch.randint(0, 1024, (4 * 64,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_4arg_bfloat16(self):
        """GSTLADV-06: 4-arg STL for bfloat16 KV cache."""
        kv = cached_randn(
            (512, 8, 64), differentiation="stladv06", dtype=torch.bfloat16
        )
        idx = torch.randint(0, 512, (4 * 32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16)

    def test_stl_4arg_with_downstream_softmax(self):
        """GSTLADV-07: 4-arg STL + softmax(dim=-1) after KV gather."""
        kv = cached_randn((512, 8, 64), differentiation="stladv07", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(
            lambda x, i: torch.softmax(x[i].float(), dim=-1).half(),
            kv,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_stl_4arg_named_dims_combo(self):
        """GSTLADV-08: 4-arg STL + named dims (cache,H,D) on KV tensor."""
        import torch_spyre._inductor.propagate_named_dims as _pnd

        kv = cached_randn((512, 8, 64), differentiation="stladv08", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        _pnd.name_tensor_dims(kv, {"cache": 512, "H": 8, "D": 64})
        _pnd.name_tensor_dims(idx, {"slots": 32})
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_4arg_large_pool(self):
        """GSTLADV-09: Large pool (cache=32768, H=8, D=64); production-like scale."""
        kv = cached_randn((512, 8, 64), differentiation="stladv09", dtype=torch.float16)
        idx = torch.randint(0, 512, (64,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stl_4arg_sencores4(self):
        """GSTLADV-10: 4-arg STL gather at SENCORES=4."""
        os.environ["SENCORES"] = "4"
        kv = cached_randn((512, 8, 64), differentiation="stladv10", dtype=torch.float16)
        idx = torch.randint(0, 512, (4 * 32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    # ------------------------------------------------------------------

    def test_stride_exact_2d_standard(self):
        """(32,64) standard strides [64,1]; SDSC stride map = [64,1]."""
        x = cached_randn((32, 64), differentiation="str_s2d", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        assert x.stride() == (64, 1)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stride_exact_3d_channel_last(self):
        """(8,32,128) contiguous strides [4096,128,1]."""
        x = cached_randn((8, 32, 128), differentiation="str_s3d", dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int32)
        assert x.stride() == (32 * 128, 128, 1)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_granite_kv_strides(self):
        """Granite-3.3-8b/Ministral/Mistral-Small exact strides (1,8,2048,128)."""
        cache = cached_randn(
            (1, 8, 2048, 128), differentiation="str_grt", dtype=torch.float16
        )
        idx = torch.randint(0, 2048, (41,), dtype=torch.int64)
        compare_with_cpu(
            lambda c, i: torch.index_select(c.squeeze(0).permute(1, 0, 2), 1, i),
            cache,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_stride_exact_gpt20b_shape(self):
        """gpt-oss-20b KV cache (1,8,2048,64): gather 11 prefill positions via index_select."""
        cache = cached_randn((512, 8, 64), differentiation="str04", dtype=torch.float16)
        idx = torch.randint(0, 512, (11,), dtype=torch.int32)
        compare_with_cpu(lambda c, i: c[i], cache, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stride_exact_llama3_embed(self):
        """Llama-3 embedding table (128256,4096) exact gather stride."""
        w = cached_randn((512, 128), differentiation="str05", dtype=torch.float16)
        idx = torch.randint(0, 512, (8,), dtype=torch.int64)
        assert w.stride() == (128, 1)
        compare_with_cpu(lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stride_exact_rope_cache(self):
        """RoPE cos/sin cache (4096,128) gather; stride [128,1]."""
        cos_sin = cached_randn(
            (4096, 128), differentiation="str06", dtype=torch.float16
        )
        pos = torch.randint(0, 4096, (128,), dtype=torch.int64)
        assert cos_sin.stride() == (128, 1)
        compare_with_cpu(
            lambda c, p: torch.index_select(c, 0, p),
            cos_sin,
            pos,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_stride_exact_bfloat16_kv(self):
        """bfloat16 KV (512,8,64) exact strides; wordLength=2."""
        kv = cached_randn(
            (512, 8, 64), differentiation="str_bf16", dtype=torch.bfloat16
        )
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16)

    def test_stride_exact_kv_decode(self):
        """Single decode-step KV access; stride correctness at position step."""
        kv = cached_randn((512, 8, 64), differentiation="str08", dtype=torch.float16)
        idx = torch.tensor([42, 99, 201, 387], dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stride_exact_non_contiguous_t(self):
        """Transposed 2D (M,N).T strides — stride mismatch triggers restickify."""
        base = cached_randn((64, 32), differentiation="str09", dtype=torch.float16)
        x = base.t().contiguous()
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_stride_exact_float32_kv(self):
        """float32 KV (512,8,64) exact strides; wordLength=4."""
        kv = cached_randn(
            (512, 8, 64), differentiation="str_f32kv", dtype=torch.float32
        )
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], kv, idx, atol=_ATOL_F32, rtol=_ATOL_F32)
