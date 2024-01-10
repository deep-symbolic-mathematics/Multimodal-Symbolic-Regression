# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import random
import argparse
import numpy as np
import torch
import os
import pickle
from pathlib import Path

import symbolicregression
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.utils import bool_flag, initialize_exp
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.envs import build_env
from symbolicregression.trainer import Trainer
from parsers import get_parser

np.seterr(all="raise")

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main(params):

    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.cpu:
        params.device = 'cuda'
        assert torch.cuda.is_available()
    else:
        params.device = 'cpu'
    symbolicregression.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    if params.eval_dump_path is None:
        params.eval_dump_path = Path(params.dump_path) / "evals_all"
        if not os.path.isdir(params.eval_dump_path):
            os.makedirs(params.eval_dump_path)
    env = build_env(params)

    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)

    # training
    if params.reload_data != "":
        data_types = [
            "valid{}".format(i) for i in range(1, len(trainer.data_path["functions"]))
        ]
    else:
        data_types = ["valid1"]

    trainer.n_equations = 0
    model_str_list = []
    z_rep_list = []
    y_list = []
    y_dist_list = []
    z_dist_list = []
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.inner_epoch = 0
        while trainer.inner_epoch < trainer.n_steps_per_epoch:

            # training steps
            for task_id in np.random.permutation(len(params.tasks)):
                task = params.tasks[task_id]
                if params.export_data:
                    trainer.export_data(task)
                else:
                    encoded_y, samples, loss = trainer.enc_dec_step(task)
                    
                trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)
        if params.debug_train_statistics:
            for task in params.tasks:
                trainer.get_generation_statistics(task)

        trainer.epoch += 1
        if trainer.epoch % 10 == 0:
            trainer.save_periodic()


if __name__ == "__main__":

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    check_model_params(params)

    # run experiment
    main(params)
