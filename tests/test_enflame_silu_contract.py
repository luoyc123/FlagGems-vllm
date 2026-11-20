"""Independent public-API regression for GCU strided MLP activations."""

import pytest
import torch

from flaggems_vllm import silu_and_mul
from flaggems_vllm.runtime import device
from flaggems_vllm.utils.codegen_config_utils import get_codegen_config

pytestmark = pytest.mark.skipif(device.vendor_name != "enflame", reason="GCU only")


@pytest.mark.parametrize("rows", [1, 32, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_silu_strided(rows, dtype):
    config = get_codegen_config()
    assert max(config.max_grid_size) <= 65535
    assert not config.prefer_block_pointer
    # Qwen3.6-27B TP2 split MLP: the views have a doubled row stride.
    torch.manual_seed(42)
    inputs = torch.randn(rows, 17408, dtype=dtype, device="gcu")
    gate, up = inputs.chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(gate.cpu().float()) * up.cpu().float()).to(
        dtype
    )
    with torch.inference_mode():
        actual = silu_and_mul(gate, up)
    torch.gcu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=0.015625, rtol=0.015625)


def test_public_silu_capture_replay():
    inputs = torch.randn(32, 17408, dtype=torch.bfloat16, device="gcu")
    gate, up = inputs.chunk(2, dim=-1)
    with torch.inference_mode():
        for _ in range(3):
            silu_and_mul(gate, up)
        torch.gcu.synchronize()
        graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(graph):
            actual = silu_and_mul(gate, up)
        for shift in (0.0, 0.25):
            inputs.add_(shift)
            graph.replay()
            torch.gcu.synchronize()
            expected = (
                torch.nn.functional.silu(gate.cpu().float()) * up.cpu().float()
            ).to(inputs.dtype)
            torch.testing.assert_close(
                actual.cpu(), expected, atol=0.015625, rtol=0.015625
            )
