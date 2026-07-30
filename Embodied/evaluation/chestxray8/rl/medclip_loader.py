"""Explicit, pinned MedCLIP-ViT loader for ChestX-ray8 rewards."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict

import torch

MEDCLIP_SOURCE = Path("/auto/data2/ykorkmaz/models/MedCLIP")
# The commit written in the experiment handoff contained one extra character.
MEDCLIP_COMMIT = "9c3396f20d5d54e4fae241b8cb06ca45848e98c9"
MEDCLIP_WEIGHTS_DIR = Path("/auto/data2/ykorkmaz/models/medclip-vit")
TEXT_BACKBONE = Path(
    "/auto/data2/ykorkmaz/models/medclip-backbones/Bio_ClinicalBERT"
)
VISION_BACKBONE = Path(
    "/auto/data2/ykorkmaz/models/medclip-backbones/swin-tiny-patch4-window7-224"
)
TEXT_REVISION = "d5892b39a4adaed74b92212a44081509db72f87b"
VISION_REVISION = "d00d478bfaf1f417d34a1186673ad4b4be264c51"

EXPECTED_HASHES = {
    MEDCLIP_WEIGHTS_DIR / "medclip-vit-pretrained.zip":
        "414b3ac515431c06018400ab2d83ee7307e314020cd4f6babbfdea4a58515725",
    MEDCLIP_WEIGHTS_DIR / "pytorch_model.bin":
        "07bf9f917ccce482ea30ec84e533a30c3ad395c424d3624f6b72edf8484f8126",
    TEXT_BACKBONE / "pytorch_model.bin":
        "a18c4c260fb5c0978b86658615106d5617050b5f14dac6ceb5e0d8beb2f9f719",
    VISION_BACKBONE / "pytorch_model.bin":
        "f2a3c1bc4ebd5f87ab22331b134c593dd24288c2dbaf7071ad49b4d9c59842d6",
    VISION_BACKBONE / "model.safetensors":
        "a941acf68a76b365d67738d6c6dce4f0db3ea5224b7cff856d3f8f3753502d29",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_medclip_artifacts() -> Dict[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=MEDCLIP_SOURCE, text=True
    ).strip()
    if commit != MEDCLIP_COMMIT:
        raise RuntimeError(f"MedCLIP source commit mismatch: {commit}")
    hashes = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual}")
        hashes[str(path)] = actual
    return hashes


def load_pinned_medclip():
    """Load MedGround-R1's exact ViT variant without implicit downloads."""
    hashes = verify_medclip_artifacts()
    # Loading medclip.__init__ also imports its dataset stack (nltk, pandas,
    # scikit-learn), none of which is needed for the reward model itself.
    package = types.ModuleType("medclip")
    package.__path__ = [str(MEDCLIP_SOURCE / "medclip")]
    sys.modules["medclip"] = package
    constants = importlib.import_module("medclip.constants")
    modeling = importlib.import_module("medclip.modeling_medclip")
    MedCLIPModel = modeling.MedCLIPModel
    MedCLIPVisionModelViT = modeling.MedCLIPVisionModelViT

    constants.BERT_TYPE = str(TEXT_BACKBONE)
    constants.VIT_TYPE = str(VISION_BACKBONE)
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    # Equivalent to MedCLIPModel.from_pretrained(input_dir=...), but avoids
    # importing the optional wget package after all artifacts are already local.
    state = torch.load(
        MEDCLIP_WEIGHTS_DIR / constants.WEIGHTS_NAME, map_location="cpu"
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"MedCLIP missing checkpoint keys: {incompatible.missing_keys}")
    allowed_unexpected = {"text_model.model.embeddings.position_ids"}
    if set(incompatible.unexpected_keys) != allowed_unexpected:
        raise RuntimeError(
            f"MedCLIP unexpected checkpoint keys: {incompatible.unexpected_keys}"
        )
    identities = {
        "text_config_name_or_path": model.text_model.model.config._name_or_path,
        "vision_config_name_or_path": model.vision_model.model.config._name_or_path,
        "text_revision": TEXT_REVISION,
        "vision_revision": VISION_REVISION,
        "source_commit": MEDCLIP_COMMIT,
        "hashes": hashes,
        "ignored_legacy_buffer": sorted(allowed_unexpected),
    }
    return model, identities
