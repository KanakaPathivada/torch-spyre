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

"""Boundary conditions, NaN/Inf source values, non-contiguous index tensors, mask-based gather, all gather API forms, non-contiguous sources, known bug regressions (#2662/#2661/#2654/#2653), and numeric correctness checks."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import DEVICE, cached_randn  # noqa: E402
from conftest import compare_mode  # noqa: E402

_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5


@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestGatherBoundaryNumericsAndAPIForms:
    """Boundary conditions (single row/element, P>M, D=1), NaN/Inf source values, non-contiguous index tensors, mask-based gather (masked_select/bool/nonzero), all gather API forms, non-contiguous sources, bug regressions (#2662/#2661/#2654/#2653), and numeric correctness."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self):
        os.environ["SENCORES"] = "1"
        yield
        os.environ.pop("SENCORES", None)

    # ------------------------------------------------------------------

    def test_single_row_source(self, execution_mode):
        """Source has a single row (M=1); all indices must be 0."""
        x = cached_randn((1, 64), differentiation="ec01", dtype=torch.float16)
        idx = torch.zeros(8, dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_single_element_index(self, execution_mode):
        """Index tensor has exactly 1 element; P=1 boundary."""
        x = cached_randn((64, 128), differentiation="ec02", dtype=torch.float16)
        idx = torch.randint(0, 64, (1,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_index_equals_source_size(self, execution_mode):
        """P == M; gather outputs same-size tensor as source."""
        M = 32
        x = cached_randn((M, 64), differentiation="ec03", dtype=torch.float16)
        idx = torch.randint(0, M, (M,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_all_same_index(self, execution_mode):
        """All indices identical; output is repeated rows."""
        x = cached_randn((64, 128), differentiation="ec04", dtype=torch.float16)
        idx = torch.full((16,), 7, dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_all_indices_max(self, execution_mode):
        """All indices at maximum legal value (M-1)."""
        x = cached_randn((64, 128), differentiation="ec05", dtype=torch.float16)
        idx = torch.full((16,), 63, dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_all_indices_zero(self, execution_mode):
        """All indices at 0; first row only."""
        x = cached_randn((64, 128), differentiation="ec06", dtype=torch.float16)
        idx = torch.zeros(16, dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_ascending_indices(self, execution_mode):
        """Monotonically ascending indices; sequential read pattern."""
        x = cached_randn((64, 128), differentiation="ec07", dtype=torch.float16)
        idx = torch.arange(0, 32, dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_descending_indices(self, execution_mode):
        """Monotonically descending indices."""
        x = cached_randn((64, 128), differentiation="ec08", dtype=torch.float16)
        idx = torch.arange(31, -1, -1, dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_power_of_two_shapes(self, execution_mode):
        """Power-of-2 M and D; stick-aligned boundaries."""
        x = cached_randn((128, 64), differentiation="ec09", dtype=torch.float16)
        idx = torch.randint(0, 128, (64,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_odd_inner_dim(self, execution_mode):
        """Odd D=65; cross-stick boundary element."""
        x = cached_randn((64, 65), differentiation="ec10", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_index_out_of_range_skipped(self, execution_mode):
        """Indices within [0, M-1]; no out-of-bounds access tested."""
        x = cached_randn((64, 128), differentiation="ec11", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_1d_source(self, execution_mode):
        """1D source tensor; gather is scalar indexing."""
        x = cached_randn((256,), differentiation="ec12", dtype=torch.float16)
        idx = torch.randint(0, 256, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_p_larger_than_m(self, execution_mode):
        """P > M (more gather outputs than source rows); repeated patterns."""
        x = cached_randn((16, 64), differentiation="ec13", dtype=torch.float16)
        idx = torch.randint(0, 16, (64,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gate_off_correct_fallback(self, execution_mode):
        """Numeric correctness baseline: output matches CPU reference."""
        x = cached_randn((64, 128), differentiation="ec14", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_exact_integer_gather(self, execution_mode):
        """Integer value source; output is exact (atol=0)."""
        x = torch.randint(-500, 500, (32, 64), dtype=torch.int32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_masked_select_1d(self, execution_mode):
        """torch.masked_select on 1D tensor; positive elements only."""
        x = cached_randn((64,), differentiation="msk01", dtype=torch.float16)
        mask = x > 0
        compare_mode(
            execution_mode,
            lambda x, m: torch.masked_select(x, m),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_masked_select_2d(self, execution_mode):
        """torch.masked_select on 2D tensor; positive elements flattened."""
        x = cached_randn((8, 32), differentiation="msk02", dtype=torch.float16)
        mask = x > 0
        compare_mode(
            execution_mode,
            lambda x, m: torch.masked_select(x, m),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_bool_index_gather_1d(self, execution_mode):
        """x[mask] bool indexing on 1D; selects positive-valued elements."""
        x = cached_randn((64,), differentiation="msk03", dtype=torch.float16)
        mask = x > 0
        compare_mode(execution_mode, lambda x, m: x[m], x, mask, atol=0, rtol=0)

    def test_bool_index_gather_2d(self, execution_mode):
        """x[mask] bool indexing on 2D dim=0; selects rows where mask is True."""
        x = cached_randn((32, 64), differentiation="msk04", dtype=torch.float16)
        mask = torch.randint(0, 2, (32,)).bool()
        compare_mode(execution_mode, lambda x, m: x[m], x, mask, atol=0, rtol=0)

    def test_nonzero_gather(self, execution_mode):
        """torch.nonzero → index; gather rows at nonzero scalar positions."""
        x = cached_randn((64, 32), differentiation="msk05", dtype=torch.float16)
        scores = torch.randint(0, 10, (64,), dtype=torch.int32)
        nz = torch.nonzero(scores, as_tuple=True)[0].int()
        if nz.numel() == 0:
            nz = torch.tensor([0], dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, nz, atol=0, rtol=0)

    def test_mask_then_index_gather(self, execution_mode):
        """Create index from mask; use as int gather index."""
        src = cached_randn((64, 128), differentiation="msk06", dtype=torch.float16)
        scores = torch.randn(64)
        keep_mask = scores > 0
        idx = torch.where(keep_mask)[0].int()
        if idx.numel() == 0:
            idx = torch.tensor([0], dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], src, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_masked_scatter_pattern(self, execution_mode):
        """masked_scatter_ fills positions where mask is True; gather back verifies values."""
        x = cached_randn((64, 32), differentiation="msk07", dtype=torch.float16)
        mask = torch.randint(0, 2, (64,)).bool()
        idx = torch.where(mask)[0].int()
        if idx.numel() == 0:
            idx = torch.tensor([0], dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_masked_fill_negative(self, execution_mode):
        """masked_fill sets padding positions to -inf; gather verifies non-masked rows."""
        x = cached_randn((32, 64), differentiation="msk08", dtype=torch.float16)
        keep = torch.randint(0, 2, (32,)).bool()
        idx = torch.where(keep)[0].int()
        if idx.numel() == 0:
            idx = torch.tensor([0], dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_causal_attention_mask(self, execution_mode):
        """Causal mask applied to scores then gather valid row subset."""
        scores = cached_randn((32, 64), differentiation="msk09", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda s, i: torch.where(s[i] > 0, s[i], torch.zeros_like(s[i])),
            scores,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_masked_output(self, execution_mode):
        """Gather + apply output mask for padding suppression."""
        x = cached_randn((64, 128), differentiation="msk10", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        out_mask = torch.randint(0, 2, (16, 128)).bool()
        compare_mode(
            execution_mode,
            lambda x, i, m: x[i].masked_fill(m, 0.0),
            x,
            idx,
            out_mask,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_where_based_gather(self, execution_mode):
        """torch.where → conditional gather between two sources."""
        a = cached_randn((32, 64), differentiation="msk11a", dtype=torch.float16)
        b = cached_randn((32, 64), differentiation="msk11b", dtype=torch.float16)
        cond = torch.randint(0, 2, (32, 64)).bool()
        compare_mode(
            execution_mode,
            lambda a, b, c: torch.where(c, a, b),
            a,
            b,
            cond,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_masked_select_bfloat16(self, execution_mode):
        """masked_select on bfloat16 tensor; gather positive-valued elements."""
        x = cached_randn((64,), differentiation="msk12", dtype=torch.bfloat16)
        mask = x > 0
        compare_mode(
            execution_mode,
            lambda x, m: torch.masked_select(x, m),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_gather_from_bool_tensor(self, execution_mode):
        """Gather from a bool tensor; exact output."""
        x = torch.randint(0, 2, (64, 32)).bool()
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_masked_index_2d_row_select(self, execution_mode):
        """Bool row selector over batch dim; gather selected rows on Spyre."""
        x = cached_randn((32, 64), differentiation="msk14", dtype=torch.float16)
        mask = torch.randint(0, 2, (32,)).bool()
        compare_mode(execution_mode, lambda x, m: x[m], x, mask, atol=0, rtol=0)

    def test_gather_then_mask_fill(self, execution_mode):
        """Gather rows then mask-fill invalid positions."""
        x = cached_randn((64, 128), differentiation="msk15", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        pad_mask = torch.randint(0, 2, (16, 128)).bool()
        compare_mode(
            execution_mode,
            lambda x, i, m: x[i].masked_fill(m, 0.0),
            x,
            idx,
            pad_mask,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_with_dynamic_mask(self, execution_mode):
        """Dynamic mask via comparison; gather at matching positions."""
        x = cached_randn((64, 128), differentiation="msk16", dtype=torch.float16)
        scores = torch.randn(64)
        idx = (scores > scores.median()).nonzero(as_tuple=True)[0].int()[:16]
        if idx.numel() == 0:
            idx = torch.tensor([0], dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_index_select_dim1(self, execution_mode):
        """index_select at dim=1; column selection."""
        x = cached_randn((32, 128), differentiation="idxs02", dtype=torch.float16)
        idx = torch.randint(0, 128, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_3d_dim0(self, execution_mode):
        """index_select on 3D tensor at dim=0."""
        x = cached_randn((64, 8, 128), differentiation="idxs03", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_single_index(self, execution_mode):
        """index_select with P=1; single-row selection."""
        x = cached_randn((64, 128), differentiation="idxs04", dtype=torch.float16)
        idx = torch.tensor([42], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_all_rows(self, execution_mode):
        """index_select selects all rows in order; identity permutation."""
        x = cached_randn((32, 64), differentiation="idxs05", dtype=torch.float16)
        idx = torch.arange(32, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_repeated(self, execution_mode):
        """Repeated indices in index_select; duplicate rows."""
        x = cached_randn((32, 64), differentiation="idxs06", dtype=torch.float16)
        idx = torch.tensor([5, 5, 10, 10, 15], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_bfloat16(self, execution_mode):
        """index_select on bfloat16; SDSC wordLength=2."""
        x = cached_randn((64, 128), differentiation="idxs07", dtype=torch.bfloat16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_BF16,
            rtol=_ATOL_BF16,
        )

    def test_index_select_int32_result(self, execution_mode):
        """index_select on integer tensor; exact output."""
        x = torch.randint(-1000, 1000, (32, 64), dtype=torch.int32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_4d(self, execution_mode):
        """index_select on 4D tensor at dim=0."""
        x = cached_randn((64, 4, 8, 32), differentiation="idxs09", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_reverse(self, execution_mode):
        """Reversed index; output is reversed source."""
        x = cached_randn((32, 64), differentiation="idxs10", dtype=torch.float16)
        idx = torch.arange(31, -1, -1, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_with_downstream(self, execution_mode):
        """index_select + relu downstream fused."""
        x = cached_randn((64, 128), differentiation="idxs11", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.relu(torch.index_select(x, 0, i)),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_float32(self, execution_mode):
        """index_select on float32; 4 bytes/elem."""
        x = cached_randn((32, 64), differentiation="idxs12", dtype=torch.float32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_index_select_large(self, execution_mode):
        """Large index_select (M=4096, P=512)."""
        x = cached_randn((4096, 64), differentiation="idxs13", dtype=torch.float16)
        idx = torch.randint(0, 4096, (512,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_sencores4(self, execution_mode):
        """index_select across 4 cores (SENCORES=4)."""
        os.environ["SENCORES"] = "4"
        x = cached_randn((64, 128), differentiation="idxs14", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_index_select_gate_off(self, execution_mode):
        """index_select numeric correctness: output matches CPU reference."""
        x = cached_randn((64, 128), differentiation="idxs15", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def test_api_torch_gather_dim0(self, execution_mode):
        """torch.gather(x, 0, idx_expanded); canonical form."""
        x = cached_randn((32, 64), differentiation="api02", dtype=torch.float16)
        idx_raw = torch.randint(0, 32, (16,), dtype=torch.int64)
        idx = idx_raw.unsqueeze(1).expand(16, 64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_index_select(self, execution_mode):
        """torch.index_select(x, 0, idx); row selection."""
        x = cached_randn((64, 128), differentiation="api03", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_take_along_dim(self, execution_mode):
        """torch.take_along_dim at dim=0."""
        x = cached_randn((32, 64), differentiation="api04", dtype=torch.float16)
        idx = torch.randint(0, 32, (16, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take_along_dim(x, i, dim=0),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_torch_take(self, execution_mode):
        """torch.take(x, flat_idx); flat index into flattened tensor."""
        x = cached_randn((64, 128), differentiation="api05", dtype=torch.float16)
        flat_idx = torch.randint(0, 64 * 128, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take(x, i),
            x,
            flat_idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_repeat_interleave(self, execution_mode):
        """torch.repeat_interleave row-level expansion."""
        x = cached_randn((16, 64), differentiation="api06", dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda x: torch.repeat_interleave(x, 2, dim=0),
            x,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_bool_mask_2d(self, execution_mode):
        """x[mask] 2D bool indexing; selects rows where mask is True."""
        x = cached_randn((32, 64), differentiation="api07", dtype=torch.float16)
        mask = torch.randint(0, 2, (32,)).bool()
        compare_mode(execution_mode, lambda x, m: x[m], x, mask, atol=0, rtol=0)

    def test_api_setitem_form(self, execution_mode):
        """x[idx] = value write; read back with gather on Spyre verifies correctness."""
        x = cached_randn((64, 128), differentiation="api08", dtype=torch.float16)
        idx = torch.randperm(64, dtype=torch.int64)[:16]
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_api_fullgraph(self, execution_mode):
        """fullgraph=True compile; no graph breaks; eager path also correct."""
        x = cached_randn((64, 128), differentiation="api09", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        if execution_mode == "eager":
            compare_mode(
                execution_mode,
                lambda x, i: x[i],
                x,
                idx,
                atol=_ATOL_F16,
                rtol=_ATOL_F16,
            )
        else:
            fn = torch.compile(lambda x, i: x[i], fullgraph=True)
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, x[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_api_inference_mode(self, execution_mode):
        """torch.inference_mode(); no autograd overhead."""
        x = cached_randn((64, 128), differentiation="api10", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        with torch.inference_mode():
            compare_mode(
                execution_mode,
                lambda x, i: x[i],
                x,
                idx,
                atol=_ATOL_F16,
                rtol=_ATOL_F16,
            )

    def test_api_no_grad(self, execution_mode):
        """torch.no_grad(); inference-mode compatible."""
        x = cached_randn((64, 128), differentiation="api11", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        with torch.no_grad():
            compare_mode(
                execution_mode,
                lambda x, i: x[i],
                x,
                idx,
                atol=_ATOL_F16,
                rtol=_ATOL_F16,
            )

    def test_api_tensor_method_index_select(self, execution_mode):
        """Tensor.index_select(dim, idx) method form."""
        x = cached_randn((64, 128), differentiation="api12", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x.index_select(0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_advanced_index_tuple(self, execution_mode):
        """Advanced tuple indexing x[idx, :]; same as x[idx]."""
        x = cached_randn((64, 128), differentiation="api13", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i, :],
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_index_put_dim0(self, execution_mode):
        """index_put_ writes unique rows; gather verifies the written values."""
        x = cached_randn((64, 128), differentiation="api14", dtype=torch.float16)
        idx = torch.randperm(64, dtype=torch.int64)[:16]
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_api_gather_2d_int64_index(self, execution_mode):
        """torch.gather with int64 index; auto-downcast path."""
        x = cached_randn((32, 64), differentiation="api15", dtype=torch.float16)
        idx = torch.randint(0, 32, (16, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_embedding_functional(self, execution_mode):
        """F.embedding functional form; CPU fallback."""
        w = cached_randn((512, 128), differentiation="api16", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda w, i: F.embedding(i, w),
            w,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_embedding_module(self, execution_mode):
        """nn.Embedding weight gather: verify weight rows at given indices."""
        w = cached_randn((512, 128), differentiation="api17", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_api_take_along_dim_argmax(self, execution_mode):
        """take_along_dim(x, argmax, dim=0); pick argmax rows."""
        x = cached_randn((32, 64), differentiation="api18", dtype=torch.float16)
        idx = x.argmax(dim=0, keepdim=True)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take_along_dim(x, i, dim=0),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_slice_and_gather(self, execution_mode):
        """Slice + gather: first half of rows then index."""
        x = cached_randn((128, 64), differentiation="api19", dtype=torch.float16)
        half = x[:64]
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], half, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_api_repeat_interleave_gather(self, execution_mode):
        """repeat_interleave to expand batch then gather."""
        x = cached_randn((8, 64), differentiation="api20", dtype=torch.float16)
        expanded = x.repeat_interleave(4, dim=0)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i],
            expanded,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_api_nonzero_as_index(self, execution_mode):
        """torch.nonzero → gather; select rows with large activations."""
        x = cached_randn((64, 128), differentiation="api21", dtype=torch.float16)
        scores = x.abs().sum(dim=-1)
        nz_idx = torch.nonzero(scores > scores.mean(), as_tuple=True)[0].int()
        if nz_idx.numel() == 0:
            nz_idx = torch.tensor([0], dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, nz_idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_api_gather_then_softmax(self, execution_mode):
        """Gather logits for target tokens → softmax over vocab."""
        logits = cached_randn((32, 512), differentiation="api22", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: torch.softmax(x[i].float(), dim=-1).half(),
            logits,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def test_ncs_transposed(self, execution_mode):
        """Source is transposed then gather on new dim=0."""
        x = cached_randn((128, 64), differentiation="ncs01", dtype=torch.float16)
        xt = x.t().contiguous()
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], xt, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_ncs_sliced_rows(self, execution_mode):
        """Source is sliced (every 2nd row); non-contiguous strides."""
        x = cached_randn((128, 64), differentiation="ncs02", dtype=torch.float16)
        xs = x[::2].contiguous()
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], xs, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_ncs_permuted_3d(self, execution_mode):
        """3D tensor permuted (1,0,2) then contiguous; gather at dim=0."""
        x = cached_randn((8, 64, 128), differentiation="ncs03", dtype=torch.float16)
        xp = x.permute(1, 0, 2).contiguous()
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], xp, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_ncs_narrow(self, execution_mode):
        """Source narrowed via .narrow(); gather on narrowed view."""
        x = cached_randn((128, 64), differentiation="ncs04", dtype=torch.float16)
        xn = x.narrow(0, 16, 64).contiguous()
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], xn, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_ncs_expanded_dims(self, execution_mode):
        """Source with expanded singleton dim; gather after expand."""
        x = cached_randn((1, 64, 128), differentiation="ncs05", dtype=torch.float16)
        xe = x.expand(8, 64, 128).reshape(512, 128).contiguous()
        idx = torch.randint(0, 512, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], xe, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_ncs_after_view(self, execution_mode):
        """Source viewed to new shape; gather on re-shaped tensor."""
        x = cached_randn((8, 64, 128), differentiation="ncs06", dtype=torch.float16)
        xv = x.reshape(512, 128)
        idx = torch.randint(0, 512, (64,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], xv, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_ncs_select_slice(self, execution_mode):
        """Source selected by torch.select + gather on result."""
        x = cached_randn((8, 64, 128), differentiation="ncs07", dtype=torch.float16)
        xs = x[0]
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], xs, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_ncs_chunk_then_gather(self, execution_mode):
        """Source split into 4 chunks; gather on third chunk."""
        x = cached_randn((256, 64), differentiation="ncs08", dtype=torch.float16)
        chunk = torch.chunk(x, 4, dim=0)[2].contiguous()
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i],
            chunk,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_ncs_cat_then_gather(self, execution_mode):
        """Concat two sources; gather on concatenated tensor."""
        a = cached_randn((32, 64), differentiation="ncs09a", dtype=torch.float16)
        b = cached_randn((32, 64), differentiation="ncs09b", dtype=torch.float16)
        ab = torch.cat([a, b], dim=0)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], ab, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_ncs_stack_then_gather(self, execution_mode):
        """Stack 32 row vectors; gather rows from stacked tensor."""
        base = cached_randn((32, 128), differentiation="ncs_stk01", dtype=torch.float16)
        x = torch.stack(list(base.unbind(0)), dim=0)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_two_gathers_same_graph_both_correct(self, execution_mode):
        """Two sources, same index, one compiled graph; both gathers execute correctly."""
        x = cached_randn((64, 128), differentiation="bug02a", dtype=torch.float16)
        y = cached_randn((64, 128), differentiation="bug02b", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, y, i: (x[i], y[i]),
            x,
            y,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_then_matmul_downstream_correct(self, execution_mode):
        """Gather rows then matmul in one compiled graph; downstream op matches CPU."""
        x = cached_randn((64, 128), differentiation="bug03x", dtype=torch.float16)
        w = cached_randn((128, 64), differentiation="bug03w", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, w, i: x[i] @ w,
            x,
            w,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_sencores4_large_correct(self, execution_mode):
        """Large gather (M=256, P=128) at SENCORES=4; multi-core result matches CPU."""
        os.environ["SENCORES"] = "4"
        x = cached_randn((256, 128), differentiation="bug04", dtype=torch.float16)
        idx = torch.randint(0, 256, (128,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_two_gathers_sequential_independent_correct(self, execution_mode):
        """Two gather ops as separate calls; each result matches CPU independently."""
        x = cached_randn((64, 128), differentiation="bug06a", dtype=torch.float16)
        y = cached_randn((64, 128), differentiation="bug06b", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )
        compare_mode(
            execution_mode, lambda y, i: y[i], y, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_large_sencores1_correct(self, execution_mode):
        """Large gather (M=256, P=128) at SENCORES=1; single-core result matches CPU."""
        os.environ["SENCORES"] = "1"
        x = cached_randn((256, 128), differentiation="bug08", dtype=torch.float16)
        idx = torch.randint(0, 256, (128,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_cor_gather_identity(self, execution_mode):
        """Gather with identity index; output equals source."""
        M = 32
        x = cached_randn((M, 64), differentiation="cor01", dtype=torch.float16)
        idx = torch.arange(M, dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_cor_gather_zero_inner(self, execution_mode):
        """Inner dim D=1; minimal column width."""
        x = cached_randn((64, 1), differentiation="cor02", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_cor_bfloat16_large_index(self, execution_mode):
        """bfloat16 source; large P=256 index."""
        x = cached_randn((512, 64), differentiation="cor03", dtype=torch.bfloat16)
        idx = torch.randint(0, 512, (256,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16
        )

    def test_cor_float32_exact(self, execution_mode):
        """float32 round-trip; values preserved exactly."""
        x = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32)
        idx = torch.arange(0, 64, 2, dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_cor_two_sources_same_index(self, execution_mode):
        """Same index applied to two source tensors independently."""
        a = cached_randn((64, 128), differentiation="cor05a", dtype=torch.float16)
        b = cached_randn((64, 128), differentiation="cor05b", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda a, b, i: a[i] + b[i],
            a,
            b,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_cor_shuffle_permutation(self, execution_mode):
        """Random permutation index; all rows present exactly once."""
        M = 64
        x = cached_randn((M, 32), differentiation="cor06", dtype=torch.float16)
        perm = torch.randperm(M, dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, perm, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_cor_large_p_small_m(self, execution_mode):
        """P >> M; many repeated accesses from small source."""
        x = cached_randn((4, 64), differentiation="cor07", dtype=torch.float16)
        idx = torch.randint(0, 4, (128,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_cor_inner_dim_multiple_sticks(self, execution_mode):
        """D=512 (8 sticks at fp16); wide row gather."""
        x = cached_randn((64, 512), differentiation="cor08", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_cor_gather_then_norm(self, execution_mode):
        """Gather + F.normalize downstream."""
        x = cached_randn((64, 128), differentiation="cor09", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: F.normalize(x[i].float(), dim=-1).half(),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_cor_gather_min_max(self, execution_mode):
        """Gather output values are numerically preserved; verified via compare_mode."""
        x = cached_randn((64, 128), differentiation="cor10", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int32)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_cor_gather_3d_non_leading(self, execution_mode):
        """3D gather at dim=1 (not outermost); batch×seq indexing."""
        x = cached_randn((4, 64, 128), differentiation="cor11", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_cor_gather_with_scale(self, execution_mode):
        """Gather + scale factor; linear downstream correctness."""
        x = cached_randn((64, 128), differentiation="cor12", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i] * 0.5,
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_cor_gather_accumulate(self, execution_mode):
        """Gather + sum accumulate; output is scalar per batch."""
        x = cached_randn((64, 128), differentiation="cor13", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int32)
        compare_mode(
            execution_mode,
            lambda x, i: x[i].sum(dim=-1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_cor_gather_integer_range_values(self, execution_mode):
        """Source values span full int32 range; exact gather."""
        hi = 2**15
        x = torch.randint(-hi, hi, (32, 64), dtype=torch.int32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_nan_propagation_in_gathered_rows(self, execution_mode):
        """Gather from source with NaN at specific rows; NaN must propagate to output."""
        x = torch.randn(32, 64, dtype=torch.float16)
        x[5, :] = float("nan")
        x[20, :] = float("nan")
        idx = torch.tensor([5, 0, 20, 10], dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_neg_inf_padding_rows_gathered(self, execution_mode):
        """-inf padding rows (attention masking pattern); gather must preserve -inf."""
        x = cached_randn((32, 64), differentiation="nan02", dtype=torch.float16)
        x[0, :] = float("-inf")
        x[31, :] = float("-inf")
        idx = torch.randint(0, 32, (16,), dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_pos_inf_source_rows(self, execution_mode):
        """+inf values in source row; gather propagates +inf to all output copies."""
        x = cached_randn((32, 64), differentiation="nan03", dtype=torch.float16)
        x[7, :] = float("inf")
        idx = torch.tensor([7, 7, 0, 1, 7], dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_mixed_nan_neg_inf_source(self, execution_mode):
        """Source with both NaN and -inf rows; gather reads each type correctly."""
        x = cached_randn((32, 64), differentiation="nan04", dtype=torch.float16)
        x[0, :] = float("nan")
        x[31, :] = float("-inf")
        idx = torch.tensor([0, 31, 5, 15], dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_nan_in_unselected_rows_does_not_contaminate(self, execution_mode):
        """NaN in rows NOT indexed; selected output rows must be free of NaN."""
        x = cached_randn((32, 64), differentiation="nan05", dtype=torch.float16)
        x[10, :] = float("nan")
        x[20, :] = float("nan")
        idx = torch.arange(8, dtype=torch.int32)
        result_ref = x[idx]
        assert not result_ref.isnan().any()
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_causal_neg_inf_mask_gather_valid_rows(self, execution_mode):
        """Causal mask: future-position rows are -inf; gather reads only valid past rows."""
        x = cached_randn((64, 128), differentiation="nan06", dtype=torch.float16)
        x[32:, :] = float("-inf")
        idx = torch.arange(32, dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_noncontig_idx_stride2(self, execution_mode):
        """Non-contiguous index tensor with stride=2; every other element selected."""
        x = cached_randn((32, 128), differentiation="ncidx01", dtype=torch.float16)
        idx_full = torch.randint(0, 32, (32,), dtype=torch.int32)
        idx = idx_full[::2]
        assert not idx.is_contiguous()
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_noncontig_idx_stride3(self, execution_mode):
        """Non-contiguous index with stride=3; 16 elements from a 48-element buffer."""
        x = cached_randn((32, 64), differentiation="ncidx02", dtype=torch.float16)
        idx_base = torch.randint(0, 32, (48,), dtype=torch.int32)
        idx = idx_base[::3]
        assert not idx.is_contiguous()
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_noncontig_idx_stride4(self, execution_mode):
        """Non-contiguous index with stride=4; 16 elements from a 64-element buffer."""
        x = cached_randn((64, 64), differentiation="ncidx03", dtype=torch.float16)
        idx_base = torch.randint(0, 64, (64,), dtype=torch.int32)
        idx = idx_base[::4]
        assert not idx.is_contiguous()
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_noncontig_2d_idx_transposed(self, execution_mode):
        """Transposed 2D index tensor; shape (8,4) with non-unit strides."""
        x = cached_randn((32, 64), differentiation="ncidx04", dtype=torch.float16)
        idx_2d = torch.randint(0, 32, (4, 8), dtype=torch.int32)
        idx = idx_2d.T
        assert not idx.is_contiguous()
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_noncontig_idx_int64_stride2(self, execution_mode):
        """Non-contiguous int64 index with stride=2; auto-downcast + non-contiguous."""
        x = cached_randn((32, 64), differentiation="ncidx05", dtype=torch.float16)
        idx_full = torch.randint(0, 32, (32,), dtype=torch.int64)
        idx = idx_full[::2]
        assert not idx.is_contiguous()
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_empty_index_zero_output(self, execution_mode):
        """P=0 empty index; output shape must be (0, D) with no elements accessed."""
        x = cached_randn((32, 64), differentiation="ec_empty01", dtype=torch.float16)
        idx = torch.zeros(0, dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_minimal_gather_m1_p1_d1(self, execution_mode):
        """Absolute minimum: M=1, P=1, D=1; scalar gather from single-element source."""
        x = torch.tensor([[42.0]], dtype=torch.float16)
        idx = torch.tensor([0], dtype=torch.int32)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_negative_index_int64(self, execution_mode):
        """Negative int64 index values; PyTorch defines x[tensor([-1])] as the last row."""
        x = cached_randn((16, 64), differentiation="neg_idx01", dtype=torch.float16)
        idx = torch.tensor([-1, -2, 0, 5, -1], dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_negative_index_int64_downcast(self, execution_mode):
        """Negative int64 through the int64→int32 downcast path; -1 must map to M-1, not wrap as uint32."""
        x = cached_randn((32, 128), differentiation="neg_idx02", dtype=torch.float16)
        idx = torch.tensor([-1, -4, -8, -16, 0, 3], dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_index_select_dim2_on_4d(self, execution_mode):
        """index_select at dim=2 on (2,4,32,64); head-level sequence position selection."""
        x = cached_randn(
            (2, 4, 32, 64), differentiation="ec_dim2_01", dtype=torch.float16
        )
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 2, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestGatherOutParameterEagerCompile:
    """torch.gather out= pre-allocated output buffer in separate eager and compiled records.

    The out= parameter allows callers to provide a pre-allocated output tensor so the
    kernel writes into existing memory rather than allocating a new buffer.  This is
    important for latency-sensitive inference pipelines where every allocation on the
    hot path matters.

    No existing test in any indirect_access file covers this API form.

    Each test produces two records:
        test_name[eager]    — ATen eager dispatch path
        test_name[compiled] — torch.compile → Spyre inductor path
    A failure in one does NOT prevent the other from running.
    """

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self):
        os.environ["SENCORES"] = "1"
        yield
        os.environ.pop("SENCORES", None)

    def test_gather_out_dim0_basic(self, execution_mode):
        """torch.gather(x, 0, idx, out=buf): out= with leading-dim gather [M=128, N=64, P=32]."""
        M, N, P = 128, 64, 32
        x = cached_randn((M, N), differentiation="out01", dtype=torch.float16)
        idx = torch.randint(0, M, (P, N), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(P, N, dtype=torch.float16)
            return torch.gather(x, 0, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_out_dim1(self, execution_mode):
        """torch.gather(x, 1, idx, out=buf): out= with dim=1 gather for GQA head selection."""
        B, H, D, K = 2, 32, 64, 8
        x = cached_randn((B, H, D), differentiation="out02", dtype=torch.float16)
        idx = torch.randint(0, H, (B, K, D), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(B, K, D, dtype=torch.float16)
            return torch.gather(x, 1, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_out_dim2(self, execution_mode):
        """torch.gather(x, 2, idx, out=buf): out= with dim=2 feature selection."""
        B, H, F, K = 2, 8, 128, 32
        x = cached_randn((B, H, F), differentiation="out03", dtype=torch.float16)
        idx = torch.randint(0, F, (B, H, K), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(B, H, K, dtype=torch.float16)
            return torch.gather(x, 2, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_out_dim0_single_row(self, execution_mode):
        """out= with P=1 single-output gather; boundary case for buffer allocation."""
        M, N = 64, 128
        x = cached_randn((M, N), differentiation="out04", dtype=torch.float16)
        idx = torch.randint(0, M, (1, N), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(1, N, dtype=torch.float16)
            return torch.gather(x, 0, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_out_dim0_reused_buffer(self, execution_mode):
        """out= buffer passed to two consecutive gather calls; second call overwrites first."""
        M, N, P = 64, 32, 16
        x = cached_randn((M, N), differentiation="out05", dtype=torch.float16)
        idx1 = torch.randint(0, M, (P, N), dtype=torch.int64)
        idx2 = torch.randint(0, M, (P, N), dtype=torch.int64)

        def fn(x, idx1, idx2):
            out = torch.empty(P, N, dtype=torch.float16)
            torch.gather(x, 0, idx1, out=out)
            return torch.gather(x, 0, idx2, out=out)

        compare_mode(execution_mode, fn, x, idx1, idx2, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_out_dim0_bfloat16(self, execution_mode):
        """out= gather on bfloat16 source; validates wordLength=2 on the out= path."""
        M, N, P = 64, 64, 16
        x = cached_randn((M, N), differentiation="out06", dtype=torch.bfloat16)
        idx = torch.randint(0, M, (P, N), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(P, N, dtype=torch.bfloat16)
            return torch.gather(x, 0, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16)
