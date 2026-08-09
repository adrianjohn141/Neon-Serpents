from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from server.trainer.learner import DqnNetwork


@pytest.mark.skipif(not Path("/app/dist-server/server/predict-bundle.js").exists(), reason="compiled server utilities are available in the Docker trainer image")
def test_server_export_matches_browser_tensorflow_q_values(tmp_path: Path) -> None:
    torch.manual_seed(42)
    network = DqnNetwork(228).eval()
    tensors = [
        {"name": name, "shape": list(value.shape), "values": value.detach().cpu().reshape(-1).tolist()}
        for name, value in network.state_dict().items()
    ]
    source = {
        "brains": [{
            "snakeId": "nova", "environmentSteps": 123, "learningSteps": 45, "epsilon": 0.2,
            "trainingSpecVersion": 3, "tensors": tensors,
        }],
    }
    source_path = tmp_path / "weights.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    subprocess.run(["node", "/app/dist-server/server/bundle-exporter.js", str(source_path), str(tmp_path)], check=True)

    observation = np.linspace(-1.0, 1.0, 228, dtype=np.float32)
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps(observation.tolist()), encoding="utf-8")
    result = subprocess.run([
        "node", "/app/dist-server/server/predict-bundle.js", str(tmp_path / "nova-v3.nsbrain.json"), str(observation_path),
    ], check=True, capture_output=True, text=True)
    browser_q = np.asarray(json.loads(result.stdout), dtype=np.float32)
    with torch.inference_mode():
        server_q = network(torch.from_numpy(observation).unsqueeze(0)).squeeze(0).numpy()
    np.testing.assert_allclose(browser_q, server_q, rtol=0, atol=1e-5)
