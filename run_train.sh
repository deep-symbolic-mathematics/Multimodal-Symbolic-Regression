#Train Encoder-Decoder model for the Numeric-to-Symbolic Generation Task of Symbolic Regression (w/ SNIP Encoder & E2E Decoder)
python train.py --reload_model_snipenc ./weights/snip-10dmax.pth --reload_model_e2edec ./weights/e2e.pth --freeze_encoder True --batch_size 64 --dump_path ./dump --max_input_dimension 10 --n_steps_per_epoch 1000 --max_epoch 100000 --exp_name snipe2e --exp_id run-test --lr 4e-5 --latent_dim 512 --save_periodic 10 --n_dec_layers 16


#Continue Training with the Encoder-Decoder Pretrainde Checkpoints for Symbolic Regression
python train.py --reload_checkpoint ./weights/snip-e2e-sr.pth --freeze_encoder True --batch_size 64 --dump_path ./dump --max_input_dimension 10 --n_steps_per_epoch 1000 --max_epoch 100000 --exp_name snipe2e --exp_id run-test2 --lr 4e-5 --latent_dim 512 --save_periodic 10 --n_dec_layers 16
