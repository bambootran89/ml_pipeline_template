"""
Unit tests for Data Wiring components.

Tests cover:
- ContextRouter input/output routing
- Nested key access
- Wiring config parsing
- Step integration
"""

import unittest
from typing import Any, Dict

from mlproject.src.pipeline.context_router import (
    ContextRouter,
    create_router_from_config,
    parse_wiring_config,
)


class TestContextRouter(unittest.TestCase):
    """Test ContextRouter functionality."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.router = ContextRouter(
            step_id="test_step",
            input_keys={"data": "source_data", "model": "upstream_model"},
            output_keys={"predictions": "test_preds", "features": "test_feats"},
        )

    def test_get_input_with_mapping(self) -> None:
        """Test input retrieval with configured mapping."""
        context: Dict[str, Any] = {
            "source_data": [1, 2, 3],
            "upstream_model": "model_obj",
        }

        result = self.router.get_input(context, "data")
        self.assertEqual(result, [1, 2, 3])

        result = self.router.get_input(context, "model")
        self.assertEqual(result, "model_obj")

    def test_get_input_with_default(self) -> None:
        """Test input retrieval with default key fallback."""
        context: Dict[str, Any] = {"default_key": "value"}

        router = ContextRouter(step_id="test", input_keys={})
        result = router.get_input(
            context, "data", default_key="default_key", required=False
        )
        self.assertEqual(result, "value")

    def test_get_input_missing_required(self) -> None:
        """Test error on missing required input."""
        context: Dict[str, Any] = {}

        with self.assertRaises(KeyError):
            self.router.get_input(context, "data", required=True)

    def test_get_input_missing_optional(self) -> None:
        """Test None returned for missing optional input."""
        context: Dict[str, Any] = {}

        result = self.router.get_input(context, "data", required=False)
        self.assertIsNone(result)

    def test_set_output_with_mapping(self) -> None:
        """Test output storage with configured mapping."""
        context: Dict[str, Any] = {}

        self.router.set_output(context, "predictions", [0.5, 0.8])
        self.assertEqual(context["test_preds"], [0.5, 0.8])

        self.router.set_output(context, "features", [[1, 2], [3, 4]])
        self.assertEqual(context["test_feats"], [[1, 2], [3, 4]])

    def test_set_output_default_key(self) -> None:
        """Test output storage with default key pattern."""
        router = ContextRouter(step_id="my_step", output_keys={})
        context: Dict[str, Any] = {}

        router.set_output(context, "result", "value")
        self.assertEqual(context["my_step_result"], "value")

    def test_nested_key_access(self) -> None:
        """Test dot notation for nested key access."""
        context: Dict[str, Any] = {"parent": {"child": {"value": 42}}}

        router = ContextRouter(
            step_id="test",
            input_keys={"nested": "parent.child.value"},
        )

        result = router.get_input(context, "nested")
        self.assertEqual(result, 42)

    def test_nested_key_missing(self) -> None:
        """Test None for missing nested key."""
        context: Dict[str, Any] = {"parent": {}}

        router = ContextRouter(
            step_id="test",
            input_keys={"nested": "parent.missing.value"},
        )

        result = router.get_input(context, "nested", required=False)
        self.assertIsNone(result)


class TestWiringConfigParsing(unittest.TestCase):
    """Test wiring configuration parsing."""

    def test_parse_full_wiring(self) -> None:
        """Test parsing complete wiring config."""
        step_config = {
            "id": "test_step",
            "type": "generic_model",
            "wiring": {
                "inputs": {
                    "data": "source_data",
                    "features": "upstream_features",
                },
                "outputs": {
                    "model": "my_model",
                    "predictions": "my_preds",
                },
            },
        }

        result = parse_wiring_config(step_config)

        self.assertEqual(result["input_keys"]["data"], "source_data")
        self.assertEqual(result["input_keys"]["features"], "upstream_features")
        self.assertEqual(result["output_keys"]["model"], "my_model")
        self.assertEqual(result["output_keys"]["predictions"], "my_preds")

    def test_parse_shorthand_keys(self) -> None:
        """Test parsing shorthand input_key/output_key."""
        step_config = {
            "id": "test_step",
            "input_key": "my_input",
            "output_key": "my_output",
        }

        result = parse_wiring_config(step_config)

        self.assertEqual(result["input_keys"]["data"], "my_input")
        self.assertEqual(result["output_keys"]["data"], "my_output")

    def test_parse_empty_wiring(self) -> None:
        """Test parsing config without wiring."""
        step_config = {"id": "test_step", "type": "trainer"}

        result = parse_wiring_config(step_config)

        self.assertEqual(result["input_keys"], {})
        self.assertEqual(result["output_keys"], {})

    def test_create_router_from_config(self) -> None:
        """Test router creation from step config."""
        step_config = {
            "wiring": {
                "inputs": {"data": "preprocessed"},
                "outputs": {"model": "trained_model"},
            }
        }

        router = create_router_from_config("test_step", step_config)

        self.assertEqual(router.step_id, "test_step")
        self.assertEqual(router.input_keys["data"], "preprocessed")
        self.assertEqual(router.output_keys["model"], "trained_model")


class TestGetAllInputsOutputs(unittest.TestCase):
    """Test bulk input/output operations."""

    def test_get_all_inputs(self) -> None:
        """Test retrieving all configured inputs."""
        router = ContextRouter(
            step_id="test",
            input_keys={"data": "source", "config": "cfg"},
        )

        context = {"source": [1, 2], "cfg": {"k": "v"}, "extra": "ignored"}

        result = router.get_all_inputs(context)

        self.assertEqual(result["data"], [1, 2])
        self.assertEqual(result["config"], {"k": "v"})
        self.assertNotIn("extra", result)

    def test_set_all_outputs(self) -> None:
        """Test storing all outputs at once."""
        router = ContextRouter(
            step_id="test",
            output_keys={"model": "my_model", "preds": "my_preds"},
        )

        context: Dict[str, Any] = {}
        outputs = {"model": "model_obj", "preds": [0.5, 0.8]}

        router.set_all_outputs(context, outputs)

        self.assertEqual(context["my_model"], "model_obj")
        self.assertEqual(context["my_preds"], [0.5, 0.8])


if __name__ == "__main__":
    unittest.main()
