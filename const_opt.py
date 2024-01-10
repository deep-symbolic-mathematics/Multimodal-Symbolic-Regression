import math, time, copy
import numpy as np
import torch
from symbolicregression.metrics import compute_metrics
import symbolicregression.model.utils_wrapper as utils_wrapper


@torch.no_grad()
def evaluate_tree(env, tree, X, y, metric):
    numexpr_fn = env.simplifier.tree_to_numexpr_fn(tree)
    y_tilde = numexpr_fn(X)[:,0]
    metrics = compute_metrics({"true": [y], "predicted": [y_tilde], "predicted_tree": [tree]}, metrics=metric)
    return metrics[metric][0]

def order_candidates(env, X, y, candidates, metric="_mse", verbose=False):
    scores = []
    for candidate in candidates:
        if metric not in candidate:
            score = evaluate_tree(env, candidate["predicted_tree"], X, y, metric)
            if math.isnan(score): 
                score = np.infty if metric.startswith("_") else -np.infty
        else:
            score = candidates[metric]
        scores.append(score)
    ordered_idx = np.argsort(scores)  
    if not metric.startswith("_"): ordered_idx=list(reversed(ordered_idx))
    candidates = [candidates[i] for i in ordered_idx]
    return candidates

def refine(env, X, y, candidates, verbose):
    refined_candidates = []
    
    ## For skeleton model
    for i, candidate in enumerate(candidates):
        candidate_skeleton, candidate_constants =  env.generator.function_to_skeleton(candidate, constants_with_idx=True)
        if "CONSTANT" in candidate_constants:
            candidates[i] = env.wrap_equation_floats(candidate_skeleton, np.random.randn(len(candidate_constants)))

    candidates = [{"refinement_type": "NoRef", "predicted_tree": candidate} for candidate in candidates]
    candidates = order_candidates(env, X, y, candidates, metric="_mse", verbose=verbose)

    ## REMOVE SKELETON DUPLICATAS
    skeleton_candidates, candidates_to_remove = {}, []
    for i, candidate in enumerate(candidates):
        skeleton_candidate, _ = env.generator.function_to_skeleton(candidate["predicted_tree"], constants_with_idx=False)
        if skeleton_candidate.infix() in skeleton_candidates:
            candidates_to_remove.append(i)
        else:
            skeleton_candidates[skeleton_candidate.infix()]=1
    # if verbose: print("Removed {}/{} skeleton duplicata".format(len(candidates_to_remove), len(candidates)))

    candidates = [candidates[i] for i in range(len(candidates)) if i not in candidates_to_remove]

    candidates_to_refine = copy.deepcopy(candidates)
    
    for candidate in candidates_to_refine:
        refinement_strategy = utils_wrapper.BFGSRefinement()
        candidate_skeleton, candidate_constants = env.generator.function_to_skeleton(candidate["predicted_tree"], constants_with_idx=True)

        refined_candidate = refinement_strategy.go(env=env, 
                                                    tree=candidate_skeleton, 
                                                    coeffs0=candidate_constants,
                                                    X=X,
                                                    y=y,
                                                    downsample=1024,
                                                    stop_after=100)
        
        if refined_candidate is not None:
            refined_candidates.append({ 
                    "refinement_type": "BFGS",
                    "predicted_tree": refined_candidate,
                    })            
    candidates.extend(refined_candidates)  
    candidates = order_candidates(env, X, y, candidates, metric="r2")

    return candidates