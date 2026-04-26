# Copyright 2026 The Torch-Spyre Authors.
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

"""
Scalar **constants** (Python int/float) with **tensor** operands on Spyre.
"""

import time
import warnings
from contextlib import contextmanager

import pytest
import torch

from utils_inductor import cached_randn, compare_with_cpu


@contextmanager
def capture_warnings():
    """Capture all warnings during the block (for FallbackWarning checks)."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list


def _assert_fallback_warning(warning_list):
    fallback_warnings = [
        w for w in warning_list if "FallbackWarning" in str(w.category)
    ]
    assert len(fallback_warnings) > 0, (
        "Expected FallbackWarning for scalar constant, but none was raised"
    )


def _compile_and_run_on_spyre(fn, x):
    compiled_fn = torch.compile(fn, backend="inductor")
    return compiled_fn(x.to("spyre"))


class TestScalarCPUFallbackDetection:
    """
    Compiled graphs with **Python scalar** constants on Spyre: expect ``FallbackWarning``
    unless suppressed. Intentionally **no** ``@pytest.mark.filterwarnings`` here.
    """

    torch.manual_seed(0xFADE)

    @pytest.mark.parametrize(
        "fn",
        [
            pytest.param(lambda x: x + 2.5, id="add"),
            pytest.param(lambda x: x * 3.0, id="mul"),
            pytest.param(lambda x: x - 1.5, id="sub"),
            pytest.param(lambda x: x / 2.0, id="div"),
        ],
    )
    def test_scalar_binary_op_triggers_fallback_warning(self, fn):
        """``tensor op Python_float`` triggers fallback warning (merged add/mul/sub/div)."""
        x = cached_randn((128, 128), dtype=torch.float16)
        with capture_warnings() as w:
            _ = _compile_and_run_on_spyre(fn, x)
        _assert_fallback_warning(w)

    def test_multiple_scalars_trigger_multiple_warnings(self):
        """Several scalar ops in one graph each participate in scalar lowering / warnings."""

        def multiple_scalars(x):
            y = x + 1.0
            y = y * 2.0
            y = y - 0.5
            return y / 3.0

        x = cached_randn((128, 128), dtype=torch.float16)
        with capture_warnings() as w:
            _ = _compile_and_run_on_spyre(multiple_scalars, x)
        _assert_fallback_warning(w)

    def test_repeated_scalar_value_caching(self):
        """Same literal scalar reused (cache-friendly); warnings still expected."""

        def repeated_scalar(x):
            y = x + 2.5
            y = y * 2.5
            return y

        x = cached_randn((128, 128), dtype=torch.float16)
        with capture_warnings() as w:
            _ = _compile_and_run_on_spyre(repeated_scalar, x)
        _assert_fallback_warning(w)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestScalarReplacementCorrectness:
    """CPU vs Spyre correctness for **tensor** inputs with **Python scalar** constants."""

    torch.manual_seed(0xBEEF)

    @pytest.mark.parametrize(
        "fn,atol,rtol",
        [
            pytest.param(lambda x: x + 2.5, 0.1, 0.1, id="add"),
            pytest.param(lambda x: x * 3.0, 0.1, 0.1, id="mul"),
            pytest.param(lambda x: x - 1.5, 0.1, 0.1, id="sub"),
            pytest.param(lambda x: x / 2.0, 1e-3, 1e-3, id="div"),
        ],
    )
    def test_scalar_binary_op_correctness(self, execution_mode, fn, atol, rtol):
        """Merged correctness for ``+ * - /`` with scalar RHS (fp16 tensor)."""
        x = cached_randn((128, 128), dtype=torch.float16)
        compare_with_cpu(
            fn,
            x,
            atol=atol,
            rtol=rtol,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_scalar_true_divide_correctness(self, execution_mode):
        """``torch.true_divide(tensor, scalar)`` (Python 3 ``/`` style)."""

        def scalar_true_div(x):
            return torch.true_divide(x, 4.0)

        x = cached_randn((128, 128), dtype=torch.float16)
        compare_with_cpu(
            scalar_true_div,
            x,
            atol=1e-3,
            rtol=1e-3,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_chained_scalar_operations_correctness(self, execution_mode):
        """Chained ``+ * - /`` with distinct scalar constants."""

        def chained_ops(x):
            y = x + 1.0
            y = y * 2.0
            y = y - 0.5
            return y / 3.0

        x = cached_randn((128, 128), dtype=torch.float16)
        compare_with_cpu(
            chained_ops,
            x,
            atol=1e-2,
            rtol=1e-2,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_tensor_tensor_add_baseline(self, execution_mode):
        """Control: **tensor + tensor** (no Python scalar); still CPU vs Spyre parity."""

        def tensor_add_tensors(x, y):
            return x + y

        x = cached_randn((64, 64), dtype=torch.float16)
        y = cached_randn((64, 64), dtype=torch.float16, differentiation=1)
        compare_with_cpu(
            tensor_add_tensors,
            x,
            y,
            atol=5e-3,
            rtol=1e-2,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.parametrize("scalar_value", [0.0, 1.0, -1.0, 0.5, 10.0, 100.0])
    def test_various_scalar_values_correctness(self, execution_mode, scalar_value):
        """``tensor + scalar`` for several scalar constants."""

        def scalar_op(x):
            return x + scalar_value

        x = cached_randn((64, 64), dtype=torch.float16)
        compare_with_cpu(
            scalar_op,
            x,
            atol=1e-3,
            rtol=1e-3,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_scalar_operations_different_dtypes(self, execution_mode, dtype):
        """Scalar constants with fp16 / fp32 tensor."""

        def scalar_ops(x):
            return (x + 1.0) * 2.0

        x = cached_randn((64, 64), dtype=dtype)
        compare_with_cpu(
            scalar_ops,
            x,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_integer_scalar_correctness(self, execution_mode):
        """Integer Python scalars with fp16 tensor."""

        def int_scalar_ops(x):
            y = x + 2
            y = y * 3
            return y

        x = cached_randn((64, 64), dtype=torch.float16)
        compare_with_cpu(
            int_scalar_ops,
            x,
            atol=0.1,
            rtol=0.1,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_mixed_int_float_scalars_correctness(self, execution_mode):
        """Mixed int and float Python scalars."""

        def mixed_scalars(x):
            y = x + 2
            y = y * 1.5
            y = y - 3
            return y / 2.0

        x = cached_randn((64, 64), dtype=torch.float16)
        compare_with_cpu(
            mixed_scalars,
            x,
            atol=1e-2,
            rtol=1e-2,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScalarReplacementPerformance:
    """Timing smoke tests around scalar-heavy compiled graphs (not strict benchmarks)."""

    torch.manual_seed(0xCAFE)

    def test_scalar_operation_performance_overhead(self):
        """Many scalar ops in a loop; prints average time; asserts a result exists."""

        def scalar_heavy_ops(x):
            for _ in range(10):
                x = x + 1.0
                x = x * 0.99
            return x

        x = cached_randn((512, 512), dtype=torch.float16).to("spyre")
        compiled_fn = torch.compile(scalar_heavy_ops, backend="inductor")
        for _ in range(3):
            _ = compiled_fn(x)

        num_iterations = 10
        start_time = time.perf_counter()
        result = None
        for _ in range(num_iterations):
            result = compiled_fn(x)
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_iterations
        print(f"\nAverage time for scalar-heavy operations: {avg_time * 1000:.2f}ms")
        assert result is not None

    @pytest.mark.parametrize("tensor_size", [64, 128, 256, 512])
    def test_scalar_performance_scaling(self, tensor_size):
        """Scalar graph time vs square tensor side length"""

        def scalar_ops(x):
            return (x + 1.0) * 2.0 - 0.5

        x = cached_randn((tensor_size, tensor_size), dtype=torch.float16).to("spyre")
        compiled_fn = torch.compile(scalar_ops, backend="inductor")
        _ = compiled_fn(x)
        start_time = time.perf_counter()
        result = compiled_fn(x)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        print(f"\nTensor size {tensor_size}x{tensor_size}: {execution_time:.3f}ms")
        assert result is not None

    def test_scalar_caching_performance_benefit(self):
        """Same literal repeated vs distinct literals; prints timings (informational)."""

        def repeated_scalar_cached(x):
            y = x + 2.5
            y = y * 2.5
            y = y - 2.5
            return y / 2.5

        def repeated_scalar_uncached(x):
            y = x + 2.5
            y = y * 2.6
            y = y - 2.7
            return y / 2.8

        x = cached_randn((256, 256), dtype=torch.float16).to("spyre")
        cached_fn = torch.compile(repeated_scalar_cached, backend="inductor")
        uncached_fn = torch.compile(repeated_scalar_uncached, backend="inductor")
        _ = cached_fn(x)
        _ = uncached_fn(x)

        start = time.perf_counter()
        for _ in range(10):
            _ = cached_fn(x)
        cached_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(10):
            _ = uncached_fn(x)
        uncached_time = time.perf_counter() - start

        print(f"\nCached scalar time: {cached_time * 1000:.2f}ms")
        print(f"Uncached scalar time: {uncached_time * 1000:.2f}ms")
        print(f"Ratio (uncached/cached): {uncached_time / cached_time:.2f}x")


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestScalarReplacementEdgeCases:
    """Boundary scalar constants with fp16 tensors."""

    torch.manual_seed(0xDEAD)

    @pytest.mark.parametrize(
        "fn,atol,rtol",
        [
            pytest.param(
                lambda x: (x + 0.0) * 1.0 - 0.0, 1e-3, 1e-3, id="zero_one_identity"
            ),
            pytest.param(
                lambda x: (x + (-2.5)) * (-1.0) / (-2.0),
                1e-2,
                1e-2,
                id="negative_scalars",
            ),
            pytest.param(lambda x: x + 1e-4, 1e-3, 1e-2, id="very_small_scalar"),
        ],
    )
    def test_scalar_edge_cases_merged(self, execution_mode, fn, atol, rtol):
        """Merged: zeros/ones, negative scalars, tiny scalar (fp16 limits)."""
        x = cached_randn((64, 64), dtype=torch.float16)
        compare_with_cpu(
            fn,
            x,
            atol=atol,
            rtol=rtol,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    def test_large_scalar_values(self, execution_mode):
        """Large scalar multiplier; small-magnitude input to limit overflow."""

        def large_scalar_ops(x):
            return x * 1000.0

        x = cached_randn((64, 64), dtype=torch.float16, scale=0.01)
        compare_with_cpu(
            large_scalar_ops,
            x,
            atol=1.0,
            rtol=1e-2,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.parametrize(
        "tensor_shape", [(1, 1), (1, 128), (128, 1), (128, 128, 128)]
    )
    def test_scalar_operations_various_shapes(self, execution_mode, tensor_shape):
        """Scalar constants with varied tensor ranks / extents."""

        def scalar_op(x):
            return x * 2.0 + 1.0

        x = cached_randn(tensor_shape, dtype=torch.float16)
        compare_with_cpu(
            scalar_op,
            x,
            atol=1e-3,
            rtol=1e-3,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )
