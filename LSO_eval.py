import torch
import numpy as np
import pandas as pd
import sympy as sp
import os, sys
import symbolicregression
import requests

from symbolicregression.envs import build_env
from symbolicregression.model import check_model_params, build_modules
from parsers import get_parser
from symbolicregression.trainer import Trainer

from collections import OrderedDict, defaultdict
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor , get_top_k_features
from symbolicregression.model.model_wrapper import ModelWrapper
import symbolicregression.model.utils_wrapper as utils_wrapper

from symbolicregression.metrics import compute_metrics
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from sklearn.model_selection import train_test_split
from pathlib import Path
from model import SNIPSymbolicRegressor
from LSO_fit import lso_fit, LSOFitNeverGrad

import time
from tqdm import tqdm
import copy


def reload_model(modules, path, requires_grad=False):
    """
    Reload a checkpoint if we find one.
    """
    if path is None:
        path = "checkpoint.pth"
    assert os.path.isfile(path)

    data = torch.load(path)

    # reload model parameters
    for k, v in modules.items():
        try:
            weights = data[k]
            v.load_state_dict(weights)
            print("load model successful")
        except RuntimeError:  # remove the 'module.'
            weights = {name.partition(".")[2]: v for name, v in data[k].items()}
            v.load_state_dict(weights)

        v.requires_grad = requires_grad

    return modules


def read_file(filename, label="target", sep=None): 
    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert X.shape[1] == feature_names.shape[0]

    return X, y, feature_names


def evaluate_pmlb_lso(
        trainer,
        params,
        target_noise=0.0,
        random_state=29910,
        verbose=False,
        save=True,
        filter_fn=None,
        logger=None,
        save_file=None,
        save_suffix="./eval_result/eval_pmlb_lso.csv",
        rescale = True):
    
        env = trainer.env
        params = params
        path = params.reload_model
        trainer.modules = reload_model(trainer.modules, path)
        model = SNIPSymbolicRegressor(params = params, env=env, modules=trainer.modules)
        model.to(params.device)
        batch_results = defaultdict(list)
        all_datasets = pd.read_csv(
            "./dataset/pmlb/pmlb/all_summary_stats.tsv",
            sep="\t",)
        regression_datasets = all_datasets[all_datasets["task"] == "regression"]
        regression_datasets = regression_datasets[
            regression_datasets["n_categorical_features"] == 0]
        problems = regression_datasets

        if filter_fn is not None:
            problems = problems[filter_fn(problems)]
            problems = problems.loc[problems['n_features']<11]
        problem_names = problems["dataset"].values.tolist()
        
        pmlb_path = "./dataset/pmlb/datasets/"  # high_dim_datasets

        feynman_problems = pd.read_csv(
            "./dataset/feynman/FeynmanEquations.csv",
            delimiter=",",)
        feynman_problems = feynman_problems[["Filename", "Formula"]].dropna().values
        feynman_formulas = {}
        for p in range(feynman_problems.shape[0]):
            feynman_formulas[
                "feynman_" + feynman_problems[p][0].replace(".", "_")
            ] = feynman_problems[p][1]
        if save:
            save_file = save_suffix
        rng = np.random.RandomState(random_state)
        pbar = tqdm(total=len(problem_names))
        first_write = True
        counter =0 
        for problem_name in problem_names:
            counter += 1
            
            print("Sample: ", counter)
            if problem_name in feynman_formulas:
                formula = feynman_formulas[problem_name]
            else:
                formula = "???"
            print("GT equation : ", formula)
            print("EQ: ", problem_name)

            X, y, _ = read_file(pmlb_path + "{}/{}.tsv.gz".format(problem_name, problem_name))
            y = np.expand_dims(y, -1)

            x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=random_state)

            scale = target_noise * np.sqrt(np.mean(np.square(y_to_fit)))
            noise = rng.normal(loc=0.0, scale=scale, size=y_to_fit.shape)
            y_to_fit += noise

            ## Scale X 
            if not isinstance(X, list):
                X = [x_to_fit]
                Y = [y_to_fit]

            scaler = utils_wrapper.StandardScaler() if rescale else None
            scale_params = {}
            if scaler is not None:
                scaled_X = []
                for i, x in enumerate(X):
                    scaled_X.append(scaler.fit_transform(x))
                    scale_params[i]=scaler.get_params()
            else:
                scaled_X = X
                
            bag_number =1 
            done_bagging = False
            bagging_threshold = 0.99 
            max_r2_zero = 0
            max_bags = 1
            while (done_bagging == False) and (bag_number <= max_bags):
                bag_number += 1
                X_scaled_to_fit = scaled_X[0]
                Y_scaled_to_fit = Y[0]

                sample_to_learn = {'X_scaled_to_fit': 0, 'Y_scaled_to_fit':0, 'x_to_fit': 0, 'y_to_fit':0,'x_to_pred':0,'y_to_pred':0}
                sample_to_learn['X_scaled_to_fit'] = [X_scaled_to_fit] 
                sample_to_learn['Y_scaled_to_fit'] = [Y_scaled_to_fit]
                sample_to_learn['x_to_fit'] = [x_to_fit] 
                sample_to_learn['y_to_fit'] = [y_to_fit]
                sample_to_learn['x_to_predict'] = [x_to_predict]
                sample_to_learn['y_to_predict'] = [y_to_predict]
                with torch.no_grad():
                    if params.lso_optimizer == "gwo":
                        batch_results = lso_fit(sample_to_learn, env, params, model,batch_results,bag_number)
                    else:
                        opt_LSO = LSOFitNeverGrad( env, params, model, sample_to_learn, batch_results, bag_number)
                        batch_results = opt_LSO.fit_func()
                    
                batch_results = pd.DataFrame.from_dict(batch_results)
                batch_results.insert(0, "problem", problem_name)
                batch_results.insert(0, "formula", formula)
                batch_results["input_dimension"] = x_to_fit.shape[1]
                batch_results["bag_number"] = bag_number
                if batch_results["r2_zero_final_fit"][0] > max_r2_zero:
                    final_results = batch_results.copy()
                    max_r2_zero = batch_results["r2_zero_final_fit"][0]

                print("R2 zero final fit: ", batch_results["r2_zero_final_fit"][0])
            
            if save:
                dir_name = os.path.dirname(save_file)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                if first_write:
                    final_results.to_csv(save_file, index=False)
                    first_write = False
                else:
                    final_results.to_csv(
                        save_file, mode="a", header=False, index=False
                    )
            batch_results = defaultdict(list)
            pbar.update(1)


def evaluate_lso_in_domain(
        trainer,
        params,
        model,
        data_type,
        task,
        verbose=True,
        ablation_to_keep=None,
        save=False,
        logger=None,
        save_file=None,
        rescale = False,
        save_suffix=None):

        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        scores = OrderedDict({"epoch": trainer.epoch})

        params = params
        embedder =model.embedder
        encoder = model.encoder
        decoder = model.decoder
        embedder.eval()
        encoder.eval()
        decoder.eval()

        env = trainer.env

        eval_size_per_gpu = params.eval_size #old
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=eval_size_per_gpu,
            input_length_modulo=params.eval_input_length_modulo,
            test_env_seed=params.test_env_seed,)
        

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,)

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            rescale=False,)

        first_write = True
        if save:
            save_file = save_suffix

        batch_before_writing_threshold = min(
            2, eval_size_per_gpu // params.batch_size_eval)
        batch_before_writing = batch_before_writing_threshold

        if ablation_to_keep is not None:
            ablation_to_keep = list(
                map(lambda x: "info_" + x, ablation_to_keep.split(",")))
        else:
            ablation_to_keep = []

        pbar = tqdm(total=eval_size_per_gpu)

        batch_results = defaultdict(list)

        for samples, _ in iterator:
            x_to_fit = samples["x_to_fit"]
            y_to_fit = samples["y_to_fit"]
            infos = samples["infos"]
            tree = samples["tree"]
                        
            #### Scale X 
            X = x_to_fit
            Y = y_to_fit
            if not isinstance(X, list):
                X = [X]
                Y = [Y]
            n_datasets = len(X)
            dstr.top_k_features = [None for _ in range(n_datasets)]
            for i in range(n_datasets):
                dstr.top_k_features[i] = get_top_k_features(X[i], Y[i], k=dstr.model.env.params.max_input_dimension)
                X[i] = X[i][:, dstr.top_k_features[i]]

            scaler = utils_wrapper.StandardScaler() if rescale else None
            scale_params = {}
            if scaler is not None:
                scaled_X = []
                for i, x in enumerate(X):
                    scaled_X.append(scaler.fit_transform(x))
                    scale_params[i]=scaler.get_params()
            else:
                scaled_X = X

            s, time_elapsed, sample_times = lso_fit(scaled_X, Y, params,env)
            print("time elapsed for sample: ", time_elapsed)
            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
            generated_tree = list(filter(lambda x: x is not None,
                        [env.idx_to_infix(s[1:], is_float=False, str_array=False)]))
            if generated_tree == []: 
                y = None
                model_str= None
                tree = None
            else:
                dstr.start_fit = time.time()
                dstr.tree = {}           
                refined_candidate = dstr.refine(scaled_X[0], Y[0], generated_tree, verbose=False)
                dstr.tree[0] = refined_candidate

            for k, v in infos.items():
                infos[k] = v.tolist()
                
            for refinement_type in dstr.retrieve_refinements_types():

                best_gen = copy.deepcopy(
                        dstr.retrieve_tree(refinement_type=refinement_type, with_infos=True)
                    )
                predicted_tree = best_gen["predicted_tree"]
                if predicted_tree is None:
                    continue
                del best_gen["predicted_tree"]
                if "metrics" in best_gen:
                    del best_gen["metrics"]

                batch_results["predicted_tree"].append(predicted_tree)
                batch_results["predicted_tree_prefix"].append(
                    predicted_tree.prefix() if predicted_tree is not None else None
                )
                for info, val in best_gen.items():
                    batch_results[info].append(val)
                    
                for k, v in infos.items():
                    batch_results["info_" + k].extend(v)
                        
                y_tilde_to_fit = dstr.predict(x_to_fit, refinement_type=refinement_type)
                results_fit = compute_metrics(
                    {
                        "true": y_to_fit,
                        "predicted": [y_tilde_to_fit],
                        "tree": tree,
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    batch_results[k + "_fit"].extend(v)
                del results_fit

                if params.prediction_sigmas is None:
                    prediction_sigmas = []
                else:
                    prediction_sigmas = [
                        float(sigma)
                        for sigma in params.prediction_sigmas.split(",")
                    ]

                for sigma in prediction_sigmas:
                    x_to_predict = samples["x_to_predict_{}".format(sigma)]
                    y_to_predict = samples["y_to_predict_{}".format(sigma)]
                    y_tilde_to_predict = dstr.predict(
                        x_to_predict, refinement_type=refinement_type
                    )
                    results_predict = compute_metrics(
                        {
                            "true": y_to_predict,
                            "predicted": [y_tilde_to_predict],
                            "tree": tree,
                            "predicted_tree": [predicted_tree],
                        },
                        metrics=params.validation_metrics,
                    )
                    for k, v in results_predict.items():
                        batch_results[k + "_predict_{}".format(sigma)].extend(v)
                    del results_predict

                batch_results["tree"].extend(tree)
                batch_results["tree_prefix"].extend([_tree.prefix() for _tree in tree])
                batch_results["time_mcts"].extend([time_elapsed])
                batch_results["sample_times"].extend([sample_times])
                
            if save:

                batch_before_writing -= 1
                if batch_before_writing <= 0:
                    batch_results = pd.DataFrame.from_dict(batch_results)
                    if first_write:
                        batch_results.to_csv(save_file, index=False)
                        first_write = False
                    else:
                        batch_results.to_csv(
                            save_file, mode="a", header=False, index=False)

                    batch_before_writing = batch_before_writing_threshold
                    batch_results = defaultdict(list)

            bs = len(x_to_fit)
            pbar.update(bs)

        try:
            df = pd.read_csv(save_file, na_filter=True)
        except:
            return
        info_columns = filter(lambda x: x.startswith("info_"), df.columns)
        df = df.drop(columns=filter(lambda x: x not in ablation_to_keep, info_columns))

        for refinement_type, df_refinement_type in df.groupby("refinement_type"):
            avg_scores = df_refinement_type.mean().to_dict()
            for k, v in avg_scores.items():
                scores[refinement_type + "|" + k] = v
            for ablation in ablation_to_keep:
                for val, df_ablation in df_refinement_type.groupby(ablation):
                    avg_scores_ablation = df_ablation.mean()
                    for k, v in avg_scores_ablation.items():
                        scores[
                            refinement_type + "|" + k + "_{}_{}".format(ablation, val)
                        ] = v            
        return scores



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    #load data:
    parser = get_parser()
    params = parser.parse_args()
    params.batch_size = 1
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    params.n_steps_per_epoch = 100
    params.max_input_dimension = 10
    params.env_base_seed = 2023
    params.n_dec_layers = 16
    # params.beam_size = 2
    # params.lso_pop_size = 50
    # params.lso_max_iteration = 80
    # params.lso_stop_r2 = 0.992

    # params.multi_gpu = False
    # params.eval_on_pmlb = False  
    # params.eval_lso_on_pmlb =True  
    # params.eval_in_domain = False 
    # params.eval_lso_in_domain = False
    params.local_rank = -1
    params.master_port = -1
    params.num_workers = 1
    # params.target_noise = 0.0
    # params.max_input_points = 200
    params.random_state = 14423
    params.max_number_bags = 10
    # params.save_results = True
    params.eval_verbose_print = True
    params.rescale = True
    # params.pmlb_data_type =  "strogatz" # #"blackbox" #"feynman" # # 
    params.n_trees_to_refine = params.beam_size

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    
    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu

    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)

    trainer = Trainer(modules, env, params)
  
    if params.eval_lso_on_pmlb:
        target_noise = params.target_noise
        random_state = params.random_state
        data_type = params.pmlb_data_type
        save = params.save_results

        if data_type == "feynman":
            filter_fn = lambda x: x["dataset"].str.contains("feynman")
        elif data_type == "strogatz":
            print("Strogatz data")
            filter_fn = lambda x: x["dataset"].str.contains("strogatz")
        else:
            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman"))

        evaluate_pmlb_lso(
            trainer,
            params,
            target_noise=target_noise,
            verbose=params.eval_verbose_print,
            random_state=random_state,
            save=save,
            filter_fn=filter_fn,
            save_file=None,
            save_suffix="./eval_result/noise/eval_{}_optimizer_{}_popsize_{}_maxiter_{}_stopr2_{}_noise_{}.csv".format(params.pmlb_data_type,
                                                                                                                        params.lso_optimizer,
                                                                                                                        params.lso_pop_size,
                                                                                                                        params.lso_max_iteration,
                                                                                                                        params.lso_stop_r2,
                                                                                                                        params.target_noise),
        )
