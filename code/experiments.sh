#! /bin/bash
python code/main.py --checkpoint-dir ~/tensorflow_checkpoints/mri_gan_redo/second --batch-size 24 --n-epochs 50 --cycle-loss-weight 5.0 --summary-freq 100 --learning-rate 2e-3
