# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch
from .embedders import LinearPointEmbedder
from .transformer import SNIP_TransformerModel, TransformerModel, SNIP_E2E_MAP

from .sklearn_wrapper import SymbolicTransformerRegressor
from .model_wrapper import ModelWrapper

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    modules["embedder"] = LinearPointEmbedder(params, env)
    env.get_length_after_batching = modules["embedder"].get_length_after_batching

    modules["encoder_y"] = SNIP_TransformerModel(
        params,
        env.float_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    # modules["encoder_f"] = SNIP_TransformerModel(
    #     params,
    #     env.equation_id2word,
    #     is_encoder=True,
    #     with_output=False,
    #     use_prior_embeddings=False,
    #     positional_embeddings=params.enc_positional_embeddings,
    # )
    modules["decoder"] = TransformerModel(
        params,
        env.equation_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )

    modules["mapper"] = SNIP_E2E_MAP(
        params,
    )

    # reload pretrained modules
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        for k, v in modules.items():
            assert k in reloaded
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])

    # log
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(
            f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
        )

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    return modules
