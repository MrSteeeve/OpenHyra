"""Explicit model presets for the four independent API experiment groups."""
from pathlib import Path
import json

from tasks.stochopia_structuring.generator import APIContractGenerator


DEFAULT_API_MODELS = Path(__file__).with_name("api_models.json")


def load_api_models(path=DEFAULT_API_MODELS):
    data = json.loads(Path(path).read_text())
    if data["schema"] != "stochopia-comparison-api-models.v1":
        raise ValueError("unsupported comparison model configuration")
    return data


def api_generator(name, *, config_path=DEFAULT_API_MODELS, max_calls=4, max_tokens=None, timeout=120.):
    config = load_api_models(config_path)
    preset = config["models"][name]
    return APIContractGenerator(**{k: preset[k] for k in ("base_url", "model", "api_key_env", "api_format")},
        request_options=preset.get("request_options", {}), group_id=name,
        required_max_tokens=preset.get("required_max_tokens"),
        model_info={**preset.get("model_info", {}), "verified_on": config["verified_on"],
                    "reasoning_policy": config["reasoning_policy"]},
        temperature=None, max_calls=max_calls, max_tokens=max_tokens, timeout=timeout)
