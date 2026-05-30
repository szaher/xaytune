import tempfile
from unittest.mock import MagicMock, patch

import torch

from xaytune.cli import main


def _make_sd():
    return {
        "layer.weight": torch.randn(4, 4),
        "layer.bias": torch.randn(4),
    }


class TestModelMergeCLI:
    def test_linear(self):
        with (
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tok,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
            patch("torch.save"),
        ):
            mock_model_inst = MagicMock()
            mock_model_inst.state_dict.return_value = _make_sd()
            mock_model.return_value = mock_model_inst
            mock_tok.return_value = MagicMock()

            with tempfile.TemporaryDirectory() as d:
                ret = main(
                    [
                        "export",
                        "model-merge",
                        "--models",
                        "model-a",
                        "model-b",
                        "--method",
                        "linear",
                        "--output",
                        d,
                    ]
                )
                assert ret == 0

    def test_slerp(self):
        with (
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tok,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
            patch("torch.save"),
        ):
            mock_model_inst = MagicMock()
            mock_model_inst.state_dict.return_value = _make_sd()
            mock_model.return_value = mock_model_inst
            mock_tok.return_value = MagicMock()

            with tempfile.TemporaryDirectory() as d:
                ret = main(
                    [
                        "export",
                        "model-merge",
                        "--models",
                        "model-a",
                        "model-b",
                        "--method",
                        "slerp",
                        "--t",
                        "0.3",
                        "--output",
                        d,
                    ]
                )
                assert ret == 0

    def test_ties(self):
        with (
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tok,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
            patch("torch.save"),
        ):
            mock_model_inst = MagicMock()
            mock_model_inst.state_dict.return_value = _make_sd()
            mock_model.return_value = mock_model_inst
            mock_tok.return_value = MagicMock()

            with tempfile.TemporaryDirectory() as d:
                ret = main(
                    [
                        "export",
                        "model-merge",
                        "--models",
                        "model-a",
                        "model-b",
                        "--method",
                        "ties",
                        "--base-model",
                        "base",
                        "--density",
                        "0.5",
                        "--output",
                        d,
                    ]
                )
                assert ret == 0

    def test_slerp_three_models_fails(self):
        ret = main(
            [
                "export",
                "model-merge",
                "--models",
                "a",
                "b",
                "c",
                "--method",
                "slerp",
                "--output",
                "/tmp/out",
            ]
        )
        assert ret == 1

    def test_ties_no_base_fails(self):
        ret = main(
            [
                "export",
                "model-merge",
                "--models",
                "a",
                "b",
                "--method",
                "ties",
                "--output",
                "/tmp/out",
            ]
        )
        assert ret == 1
