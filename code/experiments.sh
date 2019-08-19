#! /bin/bash
python code/main.py --checkpoint-dir ~/tensorflow_checkpoints/mri_gan_redo/test1 --batch-size 8 --n-epochs 50 --cycle-loss-weight 10.0 --summary-freq 100
