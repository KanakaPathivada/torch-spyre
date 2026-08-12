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

"""Shared fixtures and helpers for the indirect_access FVT test suite.

All helpers are plain functions imported explicitly by test modules.
The `env_gather` fixture is the standard env setup for indirect access tests.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import compare_with_cpu  # noqa: E402


@pytest.fixture
def env_gather():
    """Set SENCORES=1 for indirect access tests."""
    os.environ["SENCORES"] = "1"
    yield
    os.environ.pop("SENCORES", None)


def compare_mode(execution_mode, fn, *args, atol=0.1, rtol=0.1):
    """Run fn in exactly one execution mode and compare result with CPU.

    Use with @pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
    so each mode becomes a separate test record — a failure in eager does not
    prevent compiled from running and vice versa.
    """
    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


def compiled_code(fn, *args):
    """Compile fn with Inductor, run it, and return (result, code_string)."""
    from torch._inductor.utils import run_and_get_code

    compiled_fn = torch.compile(fn, backend="inductor")
    result, code_list = run_and_get_code(compiled_fn, *args)
    return result, "\n".join(code_list)
