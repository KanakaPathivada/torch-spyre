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

"""Shared fixtures and helpers for the indirect_access test suite.

All helpers are plain functions imported explicitly by test modules.
`sencores` is a parametrized fixture used only by classes that explicitly test
multi-core behaviour; request it in env_base to opt in.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import compare_with_cpu  # noqa: E402

try:
    from torch_spyre._inductor import config as _spyre_config

    _max_sencores = _spyre_config.sencores
except ImportError:
    _max_sencores = int(os.getenv("SENCORES", "32"))

# Single-core and chip-max; MULTICORE_SENCORES subset.
MULTICORE_SENCORES = (1, _max_sencores)


@pytest.fixture(params=MULTICORE_SENCORES)
def sencores(request):
    """Parametrized over single-core (1) and chip-max sencores.

    Only kicks in for classes whose env_base requests this fixture.
    """
    return request.param


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
