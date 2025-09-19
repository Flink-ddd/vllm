# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from transformers import SiglipModel, SiglipProcessor

from ....conftest import HfRunner, VllmRunner
from ...utils import check_embeddings_close

MODEL = "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"
TOKENIZER_ID = "google/siglip-base-patch16-224"


def _run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
    *,
    input_texts: list[str],
    input_images: list["Image.Image"],
) -> None:
    """
    A centralized function to run tests. It executes vLLM, then Hugging Face,
    and compares their embedding outputs.
    """
    with vllm_runner(model,
                     runner="pooling",
                     dtype=dtype,
                     enforce_eager=True,
                     trust_remote_code=True,
                     max_model_len=64,
                     tokenizer_name=TOKENIZER_ID,
                     gpu_memory_utilization=0.8) as vllm_model:
        vllm_outputs = vllm_model.embed(input_texts, images=input_images)

    with hf_runner(model, dtype=dtype) as hf_model:
        inputs = hf_model.get_inputs(input_texts, images=input_images)
        hf_inputs = hf_model.wrap_device(inputs)

        with torch.no_grad():
            if input_images and input_texts and all(input_texts):
                # Both image and text inputs
                 hf_image_embeds = hf_model.model.get_image_features(
                    pixel_values=hf_inputs["pixel_values"])
                 hf_text_embeds = hf_model.model.get_text_features(
                    input_ids=hf_inputs["input_ids"],
                    attention_mask=hf_inputs["attention_mask"])
                 hf_outputs = (hf_image_embeds.cpu().tolist() + 
                               hf_text_embeds.cpu().tolist())
            elif input_images:
                # Image-only
                hf_outputs_tensor = hf_model.model.get_image_features(
                    pixel_values=hf_inputs["pixel_values"])
                hf_outputs = hf_outputs_tensor.cpu().tolist()
            else:
                # Text-only
                hf_outputs_tensor = hf_model.model.get_text_features(
                    input_ids=hf_inputs["input_ids"],
                    attention_mask=hf_inputs["attention_mask"])
                hf_outputs = hf_outputs_tensor.cpu().tolist()

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", [MODEL])
@pytest.mark.parametrize("dtype", ["half"])
def test_text_embeddings(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
) -> None:
    """Test text embeddings against Hugging Face."""
    input_texts = [
        "a photo of a stop sign",
        "a photo of a cherry blossom",
    ]

    _run_test(hf_runner,
              vllm_runner,
              model,
              dtype,
              input_texts=input_texts,
              input_images=[])


@pytest.mark.parametrize("model", [MODEL])
@pytest.mark.parametrize("dtype", ["half"])
def test_image_embeddings(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test image embeddings against Hugging Face."""
    assets_by_name = {asset.name: asset for asset in image_assets}
    stop_sign_image = assets_by_name["stop_sign"].pil_image

    _run_test(
        hf_runner,
        vllm_runner,
        model,
        dtype,
        input_texts=[""],
        input_images=[stop_sign_image],
    )


@pytest.mark.parametrize("model", [MODEL])
@pytest.mark.parametrize("dtype", ["half"])
def test_relative_similarity(
    vllm_runner: type[VllmRunner],
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """
    Test that the relative similarity between image and text embeddings
    is logical.
    """
    assets_by_name = {asset.name: asset for asset in image_assets}
    stop_sign_image = assets_by_name["stop_sign"].pil_image
    correct_text = "a photo of a stop sign"
    incorrect_text = "a photo of a cherry blossom"

    with vllm_runner(model,
                     runner="pooling",
                     dtype=dtype,
                     enforce_eager=True,
                     trust_remote_code=True,
                     max_model_len=64,
                     tokenizer_name=TOKENIZER_ID,
                     gpu_memory_utilization=0.8) as vllm_model:

        image_embed = torch.tensor(vllm_model.embed([], [stop_sign_image])[0])
        text_embeds = torch.tensor(
            vllm_model.embed([correct_text, incorrect_text]))

        correct_text_embed = text_embeds[0]
        incorrect_text_embed = text_embeds[1]

        sim_correct = F.cosine_similarity(image_embed,
                                          correct_text_embed,
                                          dim=0)
        sim_incorrect = F.cosine_similarity(image_embed,
                                            incorrect_text_embed,
                                            dim=0)

        assert sim_correct > sim_incorrect, (
            "Model failed the sanity check: "
            "Correct text should have higher similarity than incorrect text. "
            f"Got correct_score={sim_correct.item()} vs "
            f"incorrect_score={sim_incorrect.item()}")