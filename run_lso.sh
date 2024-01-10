#Symbolic Regression Experiments - SRBecnh Datasets

###### Strogatz
# GWO
python LSO_eval.py --reload_model ./weights/snip-e2e-sr.pth --eval_lso_on_pmlb True --pmlb_data_type strogatz --target_noise 0.0 --max_input_points 200 --lso_optimizer gwo --lso_pop_size 50 --lso_max_iteration 80 --lso_stop_r2 0.99 --beam_size 2

# NeverGrad
python LSO_eval.py --reload_model ./weights/snip-e2e-sr.pth --eval_lso_on_pmlb True --pmlb_data_type strogatz --target_noise 0.0 --max_input_points 200 --lso_optimizer ngopt --lso_pop_size 50 --lso_max_iteration 80 --lso_stop_r2 0.99 --beam_size 2


###### Feynman
# GWO
python LSO_eval.py --reload_model ./weights/snip-e2e-sr.pth --eval_lso_on_pmlb True --pmlb_data_type feynman --target_noise 0.0 --max_input_points 200 --lso_optimizer gwo --lso_pop_size 50 --lso_max_iteration 80 --lso_stop_r2 0.99 --beam_size 2 --save_results True

# NeverGrad
python LSO_eval.py --reload_model ./weights/snip-e2e-sr.pth --eval_lso_on_pmlb True --pmlb_data_type feynman --target_noise 0.0 --max_input_points 200 --lso_optimizer ngopt --lso_pop_size 50 --lso_max_iteration 80 --lso_stop_r2 0.99 --beam_size 2


###### Blackbox
# GWO
python LSO_eval.py --reload_model ./weights/snip-e2e-sr.pth --eval_lso_on_pmlb True --pmlb_data_type blackbox --target_noise 0.0 --max_input_points 200 --lso_optimizer gwo --lso_pop_size 50 --lso_max_iteration 80 --lso_stop_r2 0.99 --beam_size 2 --save_results True

# NeverGrad
python LSO_eval.py --reload_model ./weights/snip-e2e-sr.pth --eval_lso_on_pmlb True --pmlb_data_type blackbox --target_noise 0.0 --max_input_points 200 --lso_optimizer ngopt --lso_pop_size 50 --lso_max_iteration 80 --lso_stop_r2 0.99 --beam_size 2
