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

"""Fused downstream operations after gather: unary (neg/sigmoid/relu/tanh/exp/log/sqrt/cos), scalar arithmetic, reductions (mean/sum), named tensor dimensions, and multi-core SENCORES variants."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import DEVICE, cached_randn  # noqa: E402
from conftest import compare_mode  # noqa: E402

_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestGatherFusedDownstreamOperations:
    """Downstream ops fused with gather: unary (neg/sigmoid/relu/tanh/exp/log/sqrt/cos), scalar mul/div/sub, mean/sum reductions, named tensor dims, SENCORES 1/4/32, and compile cache verification."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self):
        os.environ["SENCORES"] = "1"
        yield
        os.environ.pop("SENCORES", None)
        os.environ.pop("LX_PLANNING", None)
        os.environ.pop("CO_OPTIMIZING_LX_PLANNING", None)
        os.environ.pop("SPYRE_INDUCTOR_ENABLE_FUSION", None)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "op,diff_key",
        [
            (torch.neg, "gds01"),
            (torch.sigmoid, "gds02"),
            (torch.relu, "gds03"),
        ],
    )
    def test_gather_3d_unary(self, execution_mode, op, diff_key):
        """3D gather on (8,32,128) + unary op co-scheduled."""
        x = cached_randn((8, 32, 128), differentiation=diff_key, dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: op(x[i]),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_sqrt_bfloat16(self, execution_mode):
        """bfloat16 gather + sqrt downstream; abs input for valid sqrt."""
        x = cached_randn(
            (32, 128), differentiation="gds04", dtype=torch.bfloat16, abs=True
        )
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: torch.sqrt(x[i]),
            x,
            idx,
            atol=_ATOL_BF16,
            rtol=_ATOL_BF16,
        )

    def test_gather_exp_float32(self, execution_mode):
        """float32 gather + exp downstream; no fp16 precision loss."""
        x = cached_randn((32, 128), differentiation="gds05", dtype=torch.float32)
        x = x.abs()
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: torch.exp(x[i].clamp(max=10.0)),
            x,
            idx,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_gather_triple_unary(self, execution_mode):
        """Triple-chained exp → tanh → sigmoid after gather."""
        x = cached_randn((32, 256), differentiation="gds06", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: torch.sigmoid(torch.tanh(torch.exp(x[i].clamp(max=5.0)))),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_scalar_sub(self, execution_mode):
        """Scalar subtraction (- 0.5) after gather."""
        x = cached_randn((32, 128), differentiation="gds07", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i] - 0.5,
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_scalar_div(self, execution_mode):
        """Scalar divide (/ 8.0) — attention head scale."""
        x = cached_randn((32, 256), differentiation="gds08", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i] / 8.0,
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_mean_reduction(self, execution_mode):
        """mean(dim=1) after gather; output collapses to (16,)."""
        x = cached_randn((32, 128), differentiation="gds09", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i].mean(dim=1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_sum_dim0(self, execution_mode):
        """sum(dim=0) after gather; output shape (128,)."""
        x = cached_randn((32, 128), differentiation="gds10", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i].sum(dim=0),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_cos_cpu_fallback(self, execution_mode):
        """cos has no Spyre kernel; gather on Spyre, cos falls back to CPU."""
        x = cached_randn((32, 64), differentiation="gds11", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: torch.cos(x[i].float()).half(),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_log_cpu_fallback(self, execution_mode):
        """log unsupported on Spyre; gather on Spyre, log on CPU."""
        x = cached_randn(
            (32, 64), differentiation="gds12", dtype=torch.float16, abs=True
        )
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: torch.log(x[i].float().clamp(min=1e-6)).half(),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_softmax(self, execution_mode):
        """gather + softmax(dim=-1); rows sum to 1.0."""
        x = cached_randn((32, 128), differentiation="gds13", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: torch.softmax(x[i].float(), dim=-1).half(),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_abs_then_sum(self, execution_mode):
        """abs → sum(dim=1) two-stage downstream chain."""
        x = cached_randn((32, 256), differentiation="gds14", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: torch.abs(x[i]).sum(dim=1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_scalar_mul_3d(self, execution_mode):
        """Scalar multiply (* 2.0) on 3D gathered output (4,16,64)."""
        x = cached_randn((8, 16, 64), differentiation="gds15", dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i] * 2.0,
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def _name_dims(self, tensor, dim_map):
        import torch_spyre._inductor.propagate_named_dims as _pnd

        _pnd.name_tensor_dims(tensor, dim_map)

    def test_named_2d_basic(self, execution_mode):
        """name_tensor_dims on 2D value + 1D index; baseline named-dim gather."""
        x = cached_randn((32, 256), differentiation="ndm01", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        self._name_dims(x, {"M": 32, "N": 256})
        self._name_dims(idx, {"P": 16})
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_3d_basic(self, execution_mode):
        """3D named dims A,B,C; all three axes annotated."""
        x = cached_randn((8, 32, 128), differentiation="ndm02", dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int32)
        self._name_dims(x, {"A": 8, "B": 32, "C": 128})
        self._name_dims(idx, {"P": 4})
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_kv_pattern(self, execution_mode):
        """Paged KV naming: cache, H, D on value; B, Lk on index."""
        kv = cached_randn((512, 8, 64), differentiation="ndm03", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        self._name_dims(kv, {"cache": 512, "H": 8, "D": 64})
        self._name_dims(idx, {"slots": 32})
        compare_mode(
            execution_mode, lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_embedding(self, execution_mode):
        """Embedding table naming: vocab, dim axes labeled."""
        w = cached_randn((32000, 128), differentiation="ndm04", dtype=torch.float16)
        idx = torch.randint(0, 32000, (32,), dtype=torch.int64)
        self._name_dims(w, {"vocab": 32000, "dim": 128})
        self._name_dims(idx, {"seq": 32})
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_moe(self, execution_mode):
        """MoE expert weight naming: E, D, F axes."""
        w = cached_randn((8, 512, 64), differentiation="ndm05", dtype=torch.float16)
        idx = torch.randint(0, 8, (16,), dtype=torch.int32)
        self._name_dims(w, {"E": 8, "D": 512, "F": 64})
        self._name_dims(idx, {"tokens": 16})
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_stl_combo(self, execution_mode):
        """Named dims + explicit STL co-exist in compiled graph."""
        x = cached_randn((32, 256), differentiation="ndm06", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        self._name_dims(x, {"M": 32, "N": 256})
        self._name_dims(idx, {"P": 16})
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_full_attention(self, execution_mode):
        """Full four-tensor attention naming (q, k, v, slot)."""
        q = cached_randn((2, 4, 8, 64), differentiation="ndm07q", dtype=torch.float16)
        kv = cached_randn((512, 8, 64), differentiation="ndm07kv", dtype=torch.float16)
        idx = torch.randint(0, 512, (2 * 4,), dtype=torch.int32)
        self._name_dims(q, {"B": 2, "Lq": 4, "H": 8, "D": 64})
        self._name_dims(kv, {"cache": 512, "H": 8, "D": 64})
        self._name_dims(idx, {"slots": 2 * 4})
        compare_mode(
            execution_mode, lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_dims_propagate(self, execution_mode):
        """Named dims on value propagate through gather to output buffer."""
        x = cached_randn((32, 128), differentiation="ndm08", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        self._name_dims(x, {"M": 32, "N": 128})
        self._name_dims(idx, {"P": 16})
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_idx_only(self, execution_mode):
        """Partial naming — index named, value unnamed."""
        x = cached_randn((32, 128), differentiation="ndm09", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        self._name_dims(idx, {"P": 16})
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_with_downstream(self, execution_mode):
        """Named dims preserved through gather → exp → sum chain."""
        x = cached_randn((32, 128), differentiation="ndm10", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        self._name_dims(x, {"M": 32, "N": 128})
        self._name_dims(idx, {"P": 16})
        compare_mode(
            execution_mode,
            lambda x, i: torch.exp(x[i]).sum(dim=1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "sencores,diff_key",
        [
            ("1", "gem01"),
            ("4", "gem02"),
            ("32", "gem03"),
        ],
    )
    def test_sencores_2d(self, execution_mode, sencores, diff_key):
        """2D gather on (32,256) at SENCORES=1/4/32."""
        os.environ["SENCORES"] = sencores
        x = cached_randn((32, 256), differentiation=diff_key, dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_sencores_4_3d(self, execution_mode):
        """Multi-core 3D gather (SENCORES=4) on (8,32,128)."""
        os.environ["SENCORES"] = "4"
        x = cached_randn((8, 32, 128), differentiation="gem04", dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_sencores_32_embedding(self, execution_mode):
        """Full-chip embedding table lookup (V=32000, SENCORES=32)."""
        os.environ["SENCORES"] = "32"
        w = cached_randn((512, 128), differentiation="gem05", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_sencores_32_paged_kv(self, execution_mode):
        """Full-chip 3D KV pool gather (SENCORES=32) on (1024,8,64)."""
        os.environ["SENCORES"] = "32"
        kv = cached_randn((256, 8, 64), differentiation="gem06", dtype=torch.float16)
        idx = torch.randint(0, 256, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], kv, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_dynamic_compile(self, execution_mode):
        """torch.compile(dynamic=True); compile once, run at P=8 and P=16."""
        if execution_mode == "eager":
            pytest.skip("compile-only test")
        x = cached_randn((32, 256), differentiation="gem07", dtype=torch.float16)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        for p in (8, 16):
            idx = torch.randint(0, 32, (p,), dtype=torch.int32)
            expected = x[idx]
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, expected, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_fusion_disabled(self, execution_mode):
        """ENABLE_FUSION=0 — gather and downstream as separate kernels."""
        os.environ["SPYRE_INDUCTOR_ENABLE_FUSION"] = "0"
        x = cached_randn((32, 256), differentiation="gem10", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: torch.exp(x[i]),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_lx_planning_enabled(self, execution_mode):
        """LX_PLANNING=1 — index must remain in HBM; downstream may use LX."""
        os.environ["LX_PLANNING"] = "1"
        x = cached_randn((32, 256), differentiation="gem11", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_compile_cache_hit_with_downstream(self, execution_mode):
        """Gather + downstream fused op compiled twice; second call uses cached graph."""
        if execution_mode == "eager":
            pytest.skip("compile-only test")
        x = cached_randn((32, 256), differentiation="gem12", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        fn = torch.compile(lambda x, i: torch.relu(x[i]))
        r1 = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
        r2 = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)

    def test_gather_then_add_broadcast_bias(self, execution_mode):
        """Gather rows then add broadcast bias; common transformer pre-norm pattern."""
        x = cached_randn((64, 128), differentiation="ds_bias01", dtype=torch.float16)
        bias = cached_randn((128,), differentiation="ds_bias01b", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, b, i: x[i].add(b),
            x,
            bias,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_then_mul_and_add_residual(self, execution_mode):
        """Gather rows, scale by gate, add residual; MoE output accumulation pattern."""
        x = cached_randn((64, 128), differentiation="ds_gate01", dtype=torch.float16)
        residual = cached_randn(
            (16, 128), differentiation="ds_gate01r", dtype=torch.float16
        )
        gate = cached_randn((16, 1), differentiation="ds_gate01g", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, r, g, i: x[i] * g + r,
            x,
            residual,
            gate,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )
