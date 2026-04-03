# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Export PaddleDetection RT-DETR models to HuggingFace safetensors format.

Converts PaddleDet state_dict keys and weight layouts to HF RT-DETR format,
producing model.safetensors, config.json, preprocessor_config.json, and
inference.yml.
"""

import json
import os
import re
import logging

import numpy as np
import paddle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HGNetV2 architecture configs (mirrors hgnet_v2.py PPHGNetV2.arch_configs)
# ---------------------------------------------------------------------------
HGNETV2_ARCH_CONFIGS = {
    "L": {
        "stem_channels": [3, 32, 48],
        "stage_config": {
            "stage1": [48, 48, 128, 1, False, False, 3, 6],
            "stage2": [128, 96, 512, 1, True, False, 3, 6],
            "stage3": [512, 192, 1024, 3, True, True, 5, 6],
            "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
        },
    },
    "X": {
        "stem_channels": [3, 32, 64],
        "stage_config": {
            "stage1": [64, 64, 128, 1, False, False, 3, 6],
            "stage2": [128, 128, 512, 2, True, False, 3, 6],
            "stage3": [512, 256, 1024, 5, True, True, 5, 6],
            "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
        },
    },
    "H": {
        "stem_channels": [3, 48, 96],
        "stage_config": {
            "stage1": [96, 96, 192, 2, False, False, 3, 6],
            "stage2": [192, 192, 512, 3, True, False, 3, 6],
            "stage3": [512, 384, 1024, 6, True, True, 5, 6],
            "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
        },
    },
}

# HF/PaddleX convention defaults for backbone_config fields that differ from
# PaddleDetection's arch_configs.  These come from PaddleX's
# DEFAULT_BACKBONE_CONFIG in _config_rt_detr.py.
HGNETV2_HF_DEFAULTS = {
    "L": {
        "depths": [3, 4, 6, 3],
        "embedding_size": 64,
        "hidden_sizes": [256, 512, 1024, 2048],
    },
    "X": {
        "depths": [3, 5, 8, 3],
        "embedding_size": 64,
        "hidden_sizes": [256, 512, 1024, 2048],
    },
    "H": {
        "depths": [4, 6, 10, 5],
        "embedding_size": 96,
        "hidden_sizes": [384, 512, 1024, 2048],
    },
}


def _convert_paddledet_to_hf(state_dict, num_classes):
    """Convert PaddleDet RT-DETR state_dict to HF RT-DETR safetensors format.

    Uses the "old" HF naming convention (out_proj, fc1/fc2, encoder.encoder)
    which is what PaddleX's ``_apply_rt_detr_key_conversion`` expects on disk.

    Args:
        state_dict: OrderedDict of PaddleDet parameter name -> numpy array.
        num_classes: Number of detection classes (for denoising_class_embed padding).

    Returns:
        Dict of HF key -> numpy array.
    """
    hf_dict = {}
    in_proj_buffer = {}  # unique_key -> {base, wb, value}
    # Track keys produced by in_proj split so we skip them during transpose
    split_keys = set()

    for key, value in state_dict.items():
        new_key = key

        # ── Backbone: backbone.* -> model.backbone.model.* ──
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]
            new_key = new_key.replace("stem.", "embedder.")
            new_key = new_key.replace("stages.", "encoder.stages.")
            new_key = new_key.replace(".aggregation_squeeze_conv.", ".aggregation.0.")
            new_key = new_key.replace(".aggregation_excitation_conv.", ".aggregation.1.")
            new_key = re.sub(r"\.conv\.(\w+)$", r".convolution.\1", new_key)
            new_key = re.sub(r"\.bn\.(\w+)$", r".normalization.\1", new_key)
            new_key = "model.backbone.model." + new_key
            new_key = new_key.replace("._mean", ".running_mean")
            new_key = new_key.replace("._variance", ".running_var")
            hf_dict[new_key] = value
            continue

        # ── Encoder input proj: neck.input_proj.* -> model.encoder_input_proj.* ──
        if new_key.startswith("neck.input_proj."):
            new_key = new_key.replace("neck.input_proj.", "model.encoder_input_proj.")
            new_key = new_key.replace("._mean", ".running_mean")
            new_key = new_key.replace("._variance", ".running_var")
            hf_dict[new_key] = value
            continue

        # ── Encoder AIFI: neck.encoder.* -> model.encoder.encoder.* (old naming) ──
        if new_key.startswith("neck.encoder."):
            new_key = new_key.replace("neck.encoder.", "model.encoder.encoder.")

            # Self-attention in_proj -> buffer for q/k/v split
            m = re.match(r"(.*\.self_attn\.)in_proj_(weight|bias)$", new_key)
            if m:
                base, wb = m.group(1), m.group(2)
                in_proj_buffer[new_key] = {"base": base, "wb": wb, "value": value}
                continue

            # old naming: keep out_proj, use fc1/fc2
            new_key = new_key.replace(".linear1.", ".fc1.")
            new_key = new_key.replace(".linear2.", ".fc2.")
            new_key = new_key.replace(".norm1.", ".self_attn_layer_norm.")
            new_key = new_key.replace(".norm2.", ".final_layer_norm.")
            hf_dict[new_key] = value
            continue

        # ── Encoder conv blocks: neck.{fpn,pan,lateral,downsample}.* -> model.encoder.* ──
        m_neck = re.match(
            r"neck\.(fpn_blocks|pan_blocks|lateral_convs|downsample_convs)\.", new_key
        )
        if m_neck:
            new_key = new_key.replace("neck.", "model.encoder.")
            new_key = re.sub(r"\.bn\.(\w+)$", r".norm.\1", new_key)
            hf_dict[new_key] = value
            continue

        # ── Decoder input proj: transformer.input_proj.N.{conv,norm}.* ->
        #    model.decoder_input_proj.N.{0,1}.* ──
        if new_key.startswith("transformer.input_proj."):
            new_key = new_key.replace("transformer.input_proj.", "model.decoder_input_proj.")
            new_key = re.sub(r"\.conv\.(\w+)$", r".0.\1", new_key)
            new_key = re.sub(r"\.norm\.(\w+)$", r".1.\1", new_key)
            new_key = new_key.replace("._mean", ".running_mean")
            new_key = new_key.replace("._variance", ".running_var")
            hf_dict[new_key] = value
            continue

        # ── Decoder class/bbox heads ──
        if new_key.startswith("transformer.dec_score_head."):
            new_key = new_key.replace("transformer.dec_score_head.", "model.decoder.class_embed.")
            hf_dict[new_key] = value
            continue
        if new_key.startswith("transformer.dec_bbox_head."):
            new_key = new_key.replace("transformer.dec_bbox_head.", "model.decoder.bbox_embed.")
            hf_dict[new_key] = value
            continue

        # ── Decoder query_pos_head ──
        if new_key.startswith("transformer.query_pos_head."):
            new_key = new_key.replace("transformer.", "model.decoder.")
            hf_dict[new_key] = value
            continue

        # ── Decoder layers: transformer.decoder.* -> model.decoder.* ──
        if new_key.startswith("transformer.decoder."):
            new_key = new_key.replace("transformer.decoder.", "model.decoder.")

            # Self-attention in_proj -> buffer for q/k/v split
            m = re.match(r"(.*\.self_attn\.)in_proj_(weight|bias)$", new_key)
            if m:
                base, wb = m.group(1), m.group(2)
                in_proj_buffer[new_key] = {"base": base, "wb": wb, "value": value}
                continue

            # old naming: keep out_proj, use fc1/fc2, rename norms and cross_attn
            new_key = new_key.replace(".cross_attn.", ".encoder_attn.")
            new_key = new_key.replace(".linear1.", ".fc1.")
            new_key = new_key.replace(".linear2.", ".fc2.")
            new_key = new_key.replace(".norm1.", ".self_attn_layer_norm.")
            new_key = new_key.replace(".norm2.", ".encoder_attn_layer_norm.")
            new_key = new_key.replace(".norm3.", ".final_layer_norm.")
            hf_dict[new_key] = value
            continue

        # ── Encoder heads & enc_output: transformer.enc_* -> model.enc_* ──
        if new_key.startswith("transformer.enc_"):
            new_key = new_key.replace("transformer.", "model.")
            hf_dict[new_key] = value
            continue

        # ── denoising_class_embed: pad [N, dim] -> [N+1, dim] ──
        if new_key.startswith("transformer.denoising_class_embed."):
            new_key = new_key.replace("transformer.", "model.")
            if "weight" in new_key:
                pad_row = np.zeros((1, value.shape[1]), dtype=value.dtype)
                value = np.concatenate([value, pad_row], axis=0)
            hf_dict[new_key] = value
            continue

        # ── Fallback: skip non-model keys (post_process, etc.) ──
        if any(new_key.startswith(p) for p in ["post_process.", "detr_head."]):
            continue

        logger.warning("Unmapped key: %s", key)

    # ── Split in_proj into q/k/v projections ──
    for _, info in in_proj_buffer.items():
        base = info["base"]
        wb = info["wb"]
        value = info["value"]

        if wb == "weight":
            # PaddleDet in_proj_weight: [dim, 3*dim] (paddle [in, out])
            # Split along axis=1, then transpose each to [out, in] for HF
            dim = value.shape[0]
            q = value[:, :dim].T
            k = value[:, dim:2*dim].T
            v = value[:, 2*dim:].T
            q_key = base + "q_proj.weight"
            k_key = base + "k_proj.weight"
            v_key = base + "v_proj.weight"
            hf_dict[q_key] = q
            hf_dict[k_key] = k
            hf_dict[v_key] = v
            split_keys.update([q_key, k_key, v_key])
        else:
            # bias: [3*dim] -> split into 3 * [dim]
            dim = value.shape[0] // 3
            hf_dict[base + "q_proj.bias"] = value[:dim]
            hf_dict[base + "k_proj.bias"] = value[dim:2*dim]
            hf_dict[base + "v_proj.bias"] = value[2*dim:]

    # ── Transpose all 2D linear weights (except split q/k/v, already transposed) ──
    transposed = 0
    for key in list(hf_dict.keys()):
        if key in split_keys:
            continue
        v = hf_dict[key]
        if (
            v.ndim == 2
            and "bias" not in key
            and "normalization" not in key
            and "norm" not in key.split(".")[-2:]
            and "running" not in key
            and "embed" not in key.split(".")[-2]
        ):
            hf_dict[key] = np.ascontiguousarray(v.T)
            transposed += 1

    # ── BN rename for remaining keys ──
    renamed = {}
    for key in list(hf_dict.keys()):
        new_key = key.replace("._mean", ".running_mean").replace("._variance", ".running_var")
        if new_key != key:
            renamed[new_key] = hf_dict.pop(key)
    hf_dict.update(renamed)

    logger.info(
        "Converted %d PaddleDet keys -> %d HF keys (transposed %d weights)",
        len(state_dict),
        len(hf_dict),
        transposed,
    )
    return hf_dict


def _build_hf_config(config):
    """Build HF RT-DETR config.json from PaddleDetection config.

    Args:
        config: PaddleDetection config dict.

    Returns:
        Dict for config.json.
    """
    num_classes = config.get("num_classes", 80)
    hidden_dim = config.get("hidden_dim", 256)

    # Determine backbone arch
    backbone_name = config.get("DETR", {}).get("backbone", "PPHGNetV2")
    if backbone_name == "PPHGNetV2":
        backbone_cfg = config.get("PPHGNetV2", {})
    else:
        backbone_cfg = config.get(backbone_name, {})
    arch = backbone_cfg.get("arch", "L")
    return_idx = backbone_cfg.get("return_idx", [1, 2, 3])
    freeze_stem_only = backbone_cfg.get("freeze_stem_only", True)
    freeze_at = backbone_cfg.get("freeze_at", 0)
    freeze_norm = backbone_cfg.get("freeze_norm", False)
    lr_mult_list = backbone_cfg.get("lr_mult_list", [1.0, 1.0, 1.0, 1.0, 1.0])

    # Build backbone_config using HF/PaddleX default values for the L arch,
    # then overlay PaddleDetection-specific fields.
    arch_cfg = HGNETV2_ARCH_CONFIGS.get(arch, HGNETV2_ARCH_CONFIGS["L"])
    stage_config = arch_cfg["stage_config"]

    stage_in_channels = [sc[0] for sc in stage_config.values()]
    stage_mid_channels = [sc[1] for sc in stage_config.values()]
    stage_out_channels = [sc[2] for sc in stage_config.values()]
    stage_num_blocks = [sc[3] for sc in stage_config.values()]
    stage_downsample = [sc[4] for sc in stage_config.values()]
    stage_light_block = [sc[5] for sc in stage_config.values()]
    stage_kernel_size = [sc[6] for sc in stage_config.values()]
    stage_numb_of_layers = [sc[7] for sc in stage_config.values()]

    # Derive out_features / out_indices from return_idx
    out_features = [f"stage{i+1}" for i in return_idx]
    out_indices = [i + 1 for i in return_idx]

    backbone_config = {
        "arch": arch,
        # HF-convention fields (use PaddleX defaults, not derived from PaddleDet)
        "depths": HGNETV2_HF_DEFAULTS.get(arch, {}).get("depths", stage_num_blocks),
        "embedding_size": HGNETV2_HF_DEFAULTS.get(arch, {}).get("embedding_size", 64),
        "hidden_act": "relu",
        "hidden_sizes": HGNETV2_HF_DEFAULTS.get(arch, {}).get(
            "hidden_sizes", stage_out_channels),
        "initializer_range": 0.02,
        "model_type": "hgnet_v2",
        "num_channels": 3,
        "out_features": out_features,
        "out_indices": out_indices,
        # PaddleX-specific fields (drive actual model construction)
        "stage_downsample": stage_downsample,
        "stage_downsample_strides": [2, 2, 2, 2],
        "stage_in_channels": stage_in_channels,
        "stage_kernel_size": stage_kernel_size,
        "stage_light_block": stage_light_block,
        "stage_mid_channels": stage_mid_channels,
        "stage_names": ["stem"] + list(stage_config.keys()),
        "stage_num_blocks": stage_num_blocks,
        "stage_numb_of_layers": stage_numb_of_layers,
        "stage_out_channels": stage_out_channels,
        "stem_channels": arch_cfg["stem_channels"],
        "stem_strides": [2, 1, 1, 2, 1],
        "use_learnable_affine_block": False,
        "return_idx": list(return_idx),
        "freeze_stem_only": freeze_stem_only,
        "freeze_at": freeze_at,
        # Use PaddleX defaults for training-specific fields (don't affect inference)
        "freeze_norm": False,
        "lr_mult_list": HGNETV2_HF_DEFAULTS.get(arch, {}).get(
            "lr_mult_list", [0.05, 0.05, 0.1, 0.15, 0.2]),
    }

    # Encoder config
    encoder_cfg = config.get("HybridEncoder", {})
    encoder_layer_cfg = encoder_cfg.get("encoder_layer", {})

    # Transformer/decoder config
    transformer_cfg = config.get("RTDETRTransformer", {})
    feat_strides = transformer_cfg.get("feat_strides", [8, 16, 32])

    # Encoder in_channels from backbone out_channels at return_idx
    encoder_in_channels = [stage_out_channels[i] for i in return_idx]

    # Build id2label / label2id
    id2label = {}
    label2id = {}
    label_list = config.get("label_list", None)
    if label_list:
        for i, name in enumerate(label_list):
            id2label[str(i)] = name
            label2id[name] = i
    else:
        for i in range(num_classes):
            id2label[str(i)] = str(i)
            label2id[str(i)] = i

    hf_config = {
        "activation_dropout": 0.0,
        "activation_function": "silu",
        "anchor_image_size": None,
        "attention_dropout": 0.0,
        "auxiliary_loss": True,
        "backbone_config": backbone_config,
        "batch_norm_eps": 1e-05,
        "box_noise_scale": transformer_cfg.get("box_noise_scale", 1.0),
        "d_model": hidden_dim,
        "decoder_activation_function": transformer_cfg.get("activation", "relu"),
        "decoder_attention_heads": transformer_cfg.get("nhead", 8),
        "decoder_ffn_dim": transformer_cfg.get("dim_feedforward", 1024),
        "decoder_in_channels": [hidden_dim] * len(return_idx),
        "decoder_layers": transformer_cfg.get("num_decoder_layers", 6),
        "decoder_n_points": 4,
        "disable_custom_kernels": True,
        "dropout": transformer_cfg.get("dropout", 0.0),
        "encode_proj_layers": encoder_cfg.get("use_encoder_idx", [2]),
        "encoder_activation_function": encoder_layer_cfg.get("activation", "gelu"),
        "encoder_attention_heads": encoder_layer_cfg.get("nhead", 8),
        "encoder_ffn_dim": encoder_layer_cfg.get("dim_feedforward", 1024),
        "encoder_hidden_dim": hidden_dim,
        "encoder_in_channels": encoder_in_channels,
        "encoder_layers": encoder_cfg.get("num_encoder_layers", 1),
        "eos_coefficient": 0.0001,
        "eval_size": None,
        "feat_strides": feat_strides,
        "focal_loss_alpha": 0.75,
        "focal_loss_gamma": 2.0,
        "freeze_backbone_batch_norms": freeze_norm,
        "hidden_expansion": encoder_cfg.get("expansion", 1.0),
        "id2label": id2label,
        "initializer_bias_prior_prob": None,
        "initializer_range": 0.01,
        "is_encoder_decoder": True,
        "label2id": label2id,
        "label_noise_ratio": transformer_cfg.get("label_noise_ratio", 0.5),
        "layer_norm_eps": 1e-05,
        "learn_initial_query": transformer_cfg.get("learnt_init_query", False),
        "matcher_alpha": 0.25,
        "matcher_bbox_cost": 5.0,
        "matcher_class_cost": 2.0,
        "matcher_gamma": 2.0,
        "matcher_giou_cost": 2.0,
        "model_type": "rt_detr",
        "normalize_before": False,
        "num_denoising": transformer_cfg.get("num_denoising", 100),
        "num_feature_levels": transformer_cfg.get("num_levels", 3),
        "num_queries": transformer_cfg.get("num_queries", 300),
        "positional_encoding_temperature": 10000,
        "transformers_version": "5.3.0.dev0",
        "use_focal_loss": config.get("use_focal_loss", True),
        "weight_loss_bbox": 5.0,
        "weight_loss_giou": 2.0,
        "weight_loss_vfl": 1.0,
        "with_box_refine": True,
    }

    return hf_config


def _build_preprocessor_config(config):
    """Build HF preprocessor_config.json from PaddleDetection config.

    Args:
        config: PaddleDetection config dict.

    Returns:
        Dict for preprocessor_config.json.
    """
    reader_cfg = config.get("TestReader", {})
    sample_transforms = reader_cfg.get("sample_transforms", [])

    target_size = [640, 640]
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]

    for t in sample_transforms:
        if isinstance(t, dict):
            if "Resize" in t:
                ts = t["Resize"].get("target_size", target_size)
                target_size = ts if isinstance(ts, list) else [ts, ts]
            if "NormalizeImage" in t:
                norm = t["NormalizeImage"]
                mean = norm.get("mean", mean)
                std = norm.get("std", std)

    return {
        "_valid_processor_keys": [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ],
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_processor_type": "RTDetrImageProcessor",
        "image_mean": [float(v) for v in mean],
        "image_std": [float(v) for v in std],
        "rescale_factor": 1.0 / 255.0,
        "resample": 2,
        "size": {
            "height": target_size[0],
            "width": target_size[1],
        },
    }


def export_safetensors(model, config, save_dir):
    """Export a PaddleDetection RT-DETR model to HF safetensors format.

    Produces:
        save_dir/model.safetensors
        save_dir/config.json
        save_dir/preprocessor_config.json

    The inference.yml is already generated by the standard export pipeline
    and should be copied/generated separately.

    Args:
        model: PaddleDetection model in dygraph mode (eval, BN converted).
        config: PaddleDetection config dict.
        save_dir: Output directory.
    """
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError(
            "safetensors and torch are required for safetensors export. "
            "Install with: pip install safetensors torch"
        )

    os.makedirs(save_dir, exist_ok=True)

    num_classes = config.get("num_classes", 80)

    # Extract state_dict as numpy arrays
    paddle_sd = model.state_dict()
    np_sd = {}
    for key, tensor in paddle_sd.items():
        if isinstance(tensor, paddle.Tensor):
            v = tensor.numpy()
        else:
            v = np.array(tensor)
        np_sd[key] = v

    # Convert keys and weights
    hf_sd = _convert_paddledet_to_hf(np_sd, num_classes)

    # Convert to torch tensors and save
    torch_sd = {}
    for key, arr in hf_sd.items():
        torch_sd[key] = torch.tensor(arr).contiguous()

    safetensors_path = os.path.join(save_dir, "model.safetensors")
    save_file(torch_sd, safetensors_path)
    logger.info("Saved model.safetensors to %s (%d keys)", safetensors_path, len(torch_sd))

    # Generate config.json
    hf_config = _build_hf_config(config)
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)
    logger.info("Saved config.json to %s", config_path)

    # Generate preprocessor_config.json
    preprocessor_config = _build_preprocessor_config(config)
    preprocessor_path = os.path.join(save_dir, "preprocessor_config.json")
    with open(preprocessor_path, "w") as f:
        json.dump(preprocessor_config, f, indent=2)
    logger.info("Saved preprocessor_config.json to %s", preprocessor_path)

    return safetensors_path
