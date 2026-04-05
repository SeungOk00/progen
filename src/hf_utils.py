import os
from pathlib import Path

from tokenizers import Tokenizer

from models.progen.modeling_progen import ProGenForCausalLM


def _find_env_file(start_dir: str | None = None) -> Path | None:
    base_dir = Path(start_dir or os.getcwd()).resolve()
    for directory in [base_dir, *base_dir.parents]:
        env_path = directory / ".env"
        if env_path.exists():
            return env_path
    return None


def load_env_file(start_dir: str | None = None) -> Path | None:
    env_path = _find_env_file(start_dir)
    if env_path is None:
        return None

    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
    return env_path


def get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def configure_hf_auth(start_dir: str | None = None) -> Path | None:
    env_path = load_env_file(start_dir)
    token = get_hf_token()
    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HF_TOKEN", token)
    return env_path


def load_model(model_name_or_path: str, device: str | None = None) -> ProGenForCausalLM:
    configure_hf_auth()
    token = get_hf_token()
    model = ProGenForCausalLM.from_pretrained(model_name_or_path, token=token)
    if device is not None:
        model = model.to(device)
    return model


def load_hf_tokenizer(model_name_or_path: str) -> Tokenizer:
    configure_hf_auth()
    token = get_hf_token()
    return Tokenizer.from_pretrained(model_name_or_path, token=token)
