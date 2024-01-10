from torch import nn
import torch.nn.functional as F
import torch
import sys 
import copy
import symbolicregression
from symbolicregression.model.transformer import TransformerModel
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.utils import bool_flag, initialize_exp
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.envs import build_env
from parsers import get_parser
from symbolicregression.utils import to_cuda
import os
import numpy as np
from pathlib import Path
from symbolicregression.trainer import Trainer
from collections import OrderedDict, defaultdict
import sympy as sp
from symbolicregression.model.model_wrapper import ModelWrapper
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor




class SNIPSymbolicRegressor(nn.Module):
    def __init__(self, params, env, modules):
        super().__init__()
        self.modules = modules
        self.params = params
        self.env = env

        self.embedder, self.encoder_y, self.mapper, self.decoder = (
            self.modules["embedder"],
            self.modules["encoder_y"],
            self.modules["mapper"],
            self.modules["decoder"],
        )
        self.embedder.eval()
        self.encoder_y.eval()
        self.mapper.eval()
        self.decoder.eval()


    def generate_from_latent(self, z_latent):
        mapped_src_enc = self.mapper(z_latent)

        generations, gen_len = self.decoder.generate_from_latent(
                mapped_src_enc,
                sample_temperature=None,
                max_len=self.params.max_target_len,)
        generations = generations.transpose(0,1)

        return generations, gen_len 


    def generate_from_latent_direct(self, z_latent):
        mapped_src_enc = self.mapper(z_latent)

        bs = mapped_src_enc.shape[0]
        num_samples = 10
        src_enc_b = mapped_src_enc
        encoded = (
            src_enc_b.unsqueeze(1)
            .expand((bs, num_samples) + src_enc_b.shape[1:])
            .contiguous()
            .view((bs * num_samples,) + src_enc_b.shape[1:])
        )
        sampling_generations, gen_len = self.decoder.generate_from_latent(
            encoded,
            sample_temperature=self.params.beam_temperature,
            max_len=self.params.max_generated_output_len,
        )
        generations = sampling_generations.transpose(0,1)

        return generations, gen_len
    
    
    def generate_from_latent_sampling(self, z_latent):
        mapped_src_enc = self.mapper(z_latent)

        bs = mapped_src_enc.shape[0]
        num_samples = self.params.beam_size
        
        src_enc_b = mapped_src_enc
        encoded = (
            src_enc_b.unsqueeze(1)
            .expand((bs, num_samples) + src_enc_b.shape[1:])
            .contiguous()
            .view((bs * num_samples,) + src_enc_b.shape[1:]))
        sampling_generations, _ = self.decoder.generate_from_latent(
            encoded,
            sample_temperature=self.params.beam_temperature,
            max_len=self.params.max_generated_output_len,)
        generations = sampling_generations.transpose(0,1)

        return generations


    def forward(self,samples,max_len):
        x1 = []
        for seq_id in range(len(samples["X_scaled_to_fit"])):
            x1.append([])
            for seq_l in range(len(samples["X_scaled_to_fit"][seq_id])):
                x1[seq_id].append([samples["X_scaled_to_fit"][seq_id][seq_l], samples["Y_scaled_to_fit"][seq_id][seq_l]])
        
        x1, len1 = self.embedder(x1)
        encoded_y = self.encoder_y("fwd", x=x1, lengths=len1, causal=False)
        generations, gen_len = self.generate_from_latent_direct(encoded_y)
        outputs = (encoded_y, generations, gen_len)
        return outputs