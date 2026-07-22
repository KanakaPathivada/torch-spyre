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

"""2D–4D gather shapes, diverse (M,N) sizes, all index access patterns (ascending/descending/strided/hotspot/random/repeated/2D), and PyTorch gather API correctness."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import cached_randn, compare_with_cpu

_ENABLE_FLAG = "SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS"
_ATOL_F16 = 1e-2


class TestGatherShapeRankAndIndexPatterns:
    """2D/3D/4D shapes, diverse (M,N) sizes, non-power-of-2 dims, all index patterns (full-range/ascending/descending/repeated/interleaved/strided/hotspot), and torch.gather/index_select/take_along_dim API correctness."""

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

    @pytest.mark.parametrize(
        "shape,P,diff_key",
        [
            ((64, 64), 32, "gfc01"),
            ((16, 1024), 8, "gfc02"),
            ((8, 4096), 4, "gfc03"),
            ((256, 64), 128, "gfc04"),
            ((4, 16), 2, "gfc05"),
        ],
    )
    def test_gather_2d_basic_shapes(self, shape, P, diff_key):
        """Basic 2D gather across diverse (M, N) shapes and output sizes."""
        x = cached_randn(shape, differentiation=diff_key, dtype=torch.float16)
        idx = torch.randint(0, shape[0], (P,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_2d_non_power_of_2_inner(self):
        """N=100 not a multiple of 64; stick padding must not leak."""
        x = cached_randn((16, 100), differentiation="gfc06", dtype=torch.float16)
        idx = torch.randint(0, 16, (8,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_2d_full_range(self):
        """Index = full row permutation; every row accessed exactly once."""
        x = cached_randn((32, 64), differentiation="gfc07", dtype=torch.float16)
        idx = torch.randperm(32, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_2d_all_same_row(self):
        """All idx values = 0; repeated reads of row 0."""
        x = cached_randn((32, 64), differentiation="gfc08", dtype=torch.float16)
        idx = torch.zeros(16, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_gather_2d_ascending_idx(self):
        """Sequential ascending idx [0..15]; cache-friendly access."""
        x = cached_randn((32, 128), differentiation="gfc09", dtype=torch.float16)
        idx = torch.arange(16, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_2d_descending_idx(self):
        """Descending idx [31..16]; reverse access order."""
        x = cached_randn((32, 128), differentiation="gfc10", dtype=torch.float16)
        idx = torch.arange(31, 15, -1, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_2d_repeated_idx(self):
        """idx has repeated values; duplicate output rows must be identical."""
        x = cached_randn((32, 64), differentiation="gfc11", dtype=torch.float16)
        idx = torch.tensor(
            [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9],
            dtype=torch.int32,
        )
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_gather_3d_wide(self):
        """3D tensor (8,32,512), dim=0 gather, P=16."""
        x = cached_randn((8, 32, 512), differentiation="gfc12", dtype=torch.float16)
        idx = torch.randint(0, 8, (16,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_3d_small(self):
        """Smallest valid 3D tensor (4,4,16), sub-stick inner dims."""
        x = cached_randn((4, 4, 16), differentiation="gfc13", dtype=torch.float16)
        idx = torch.randint(0, 4, (2,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_3d_interior_dim(self):
        """x[:, idx, :] — interior dim=1 gather, (8,16,64)."""
        x = cached_randn((8, 16, 64), differentiation="gfc14", dtype=torch.float16)
        idx = torch.randint(0, 16, (8,), dtype=torch.int64)
        compare_with_cpu(
            lambda x, i: torch.index_select(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_4d_data(self):
        """4D tensor (2,8,4,64), gather at dim=0, P=4."""
        x = cached_randn((2, 8, 4, 64), differentiation="gfc15", dtype=torch.float16)
        idx = torch.randint(0, 2, (4,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_index_select_new_shape(self):
        """torch.index_select(x,0,idx) on (16,256) with P=12."""
        x = cached_randn((16, 256), differentiation="gfc16", dtype=torch.float16)
        idx = torch.randint(0, 16, (12,), dtype=torch.int64)
        compare_with_cpu(
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_take_along_dim(self):
        """torch.take_along_dim(x, idx, dim=0), same-rank index (16,64)."""
        x = cached_randn((32, 64), differentiation="gfc17", dtype=torch.float16)
        idx = torch.randint(0, 32, (16, 64), dtype=torch.int64)
        compare_with_cpu(
            lambda x, i: torch.take_along_dim(x, i, dim=0),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_torch_gather_dim0(self):
        """torch.gather(x, 0, idx) functional form; same-rank (16,64) index."""
        x = cached_randn((32, 64), differentiation="gfc18", dtype=torch.float16)
        idx = torch.randint(0, 32, (16, 64), dtype=torch.int64)
        compare_with_cpu(
            lambda x, i: torch.gather(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_result_exact(self):
        """Integer-representable fp16 values (powers of 2); no rounding at output."""
        vals = [float(2**k) for k in range(64)]
        x = torch.tensor([vals] * 16, dtype=torch.float16)
        idx = torch.randint(0, 16, (8,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_gather_2d_interleaved_idx(self):
        """Interleaved idx [0,16,1,17,...]; alternating low/high row access."""
        x = cached_randn((32, 128), differentiation="gfc20", dtype=torch.float16)
        idx = torch.tensor(
            [v for pair in zip(range(16), range(16, 32)) for v in pair],
            dtype=torch.int32,
        )
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    # ------------------------------------------------------------------

    def test_idx_random_permutation(self):
        """Every row accessed exactly once via random permutation."""
        x = cached_randn((32, 128), differentiation="gidx01", dtype=torch.float16)
        idx = torch.randperm(32, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_hotspot_80pct(self):
        """80% of idx values point to the same row (heavy hotspot)."""
        x = cached_randn((32, 128), differentiation="gidx02", dtype=torch.float16)
        hot = [5] * 19 + [torch.randint(0, 32, (1,)).item() for _ in range(5)]
        idx = torch.tensor(hot[:24], dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_alternating(self):
        """Alternates [0,1,0,1,...]; only 2 unique rows."""
        x = cached_randn((32, 64), differentiation="gidx03", dtype=torch.float16)
        idx = torch.tensor([0, 1] * 8, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_idx_strided(self):
        """Strided access [0,4,8,...]; stride=4 row selection."""
        x = cached_randn((64, 64), differentiation="gidx04", dtype=torch.float16)
        idx = torch.arange(0, 64, 4, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_near_full(self):
        """P=M-1=31; all rows except row 31."""
        x = cached_randn((32, 64), differentiation="gidx05", dtype=torch.float16)
        idx = torch.arange(31, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_oversize(self):
        """P=32 > M=16; each input row read multiple times."""
        x = cached_randn((16, 64), differentiation="gidx06", dtype=torch.float16)
        idx = torch.randint(0, 16, (32,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_two_elements(self):
        """Minimal index P=2; smallest valid gather output."""
        x = cached_randn((32, 128), differentiation="gidx07", dtype=torch.float16)
        idx = torch.randint(0, 32, (2,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_single_row(self):
        """P=1 — single-element index; validates compile succeeds."""
        x = cached_randn((32, 256), differentiation="gidx08", dtype=torch.float16)
        idx = torch.randint(0, 32, (1,), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_full_range(self):
        """Sequential [0..M-1]; identity-permutation gather."""
        x = cached_randn((32, 64), differentiation="gidx09", dtype=torch.float16)
        idx = torch.arange(32, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_idx_2d_shape(self):
        """2D-shaped index (8,4) on (16,64) value tensor."""
        x = cached_randn((16, 64), differentiation="gidx10", dtype=torch.float16)
        idx = torch.randint(0, 16, (8, 4), dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_sorted_ascending(self):
        """Sorted ascending [0..P-1]; cache-friendly pattern."""
        x = cached_randn((64, 128), differentiation="gidx11", dtype=torch.float16)
        idx = torch.arange(32, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_sorted_descending(self):
        """Reverse-sorted [P-1..0]; reverse cache pattern."""
        x = cached_randn((64, 128), differentiation="gidx12", dtype=torch.float16)
        idx = torch.arange(31, -1, -1, dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_boundary_values(self):
        """Both boundary rows (0 and M-1=31) plus mid values."""
        x = cached_randn((32, 64), differentiation="gidx13", dtype=torch.float16)
        idx = torch.tensor(
            [0, 31, 5, 10, 15, 20, 25, 3, 8, 13, 18, 23, 28, 1, 16, 30],
            dtype=torch.int32,
        )
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_idx_repeated_pairs(self):
        """Each row appears twice consecutively [0,0,1,1,2,2,...]."""
        x = cached_randn((16, 64), differentiation="gidx14", dtype=torch.float16)
        idx = torch.tensor([v for v in range(8) for _ in range(2)], dtype=torch.int32)
        compare_with_cpu(lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_torch_gather_dim1(self):
        """torch.gather(x, 1, idx_expanded) at dim=1; selects columns per batch row."""
        x = cached_randn((8, 32, 64), differentiation="gfc_gd1", dtype=torch.float16)
        idx_raw = torch.randint(0, 32, (8, 16), dtype=torch.int64)
        idx = idx_raw.unsqueeze(-1).expand(8, 16, 64)
        compare_with_cpu(
            lambda x, i: torch.gather(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_take_along_dim_dim1(self):
        """torch.take_along_dim at dim=1; selects sequence positions per batch head."""
        x = cached_randn((4, 32, 64), differentiation="gfc_tad1", dtype=torch.float16)
        idx = torch.randint(0, 32, (4, 16, 64), dtype=torch.int64)
        compare_with_cpu(
            lambda x, i: torch.take_along_dim(x, i, dim=1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )
