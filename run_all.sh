#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper.  CPU only.
# Rough cost on 4 cores: checks ~15 min, training ~65 + 3 x 20-40 min, evaluation ~60 min.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
T=${THREADS:-4}

# -- 1. checks that need no training ---------------------------------------------------------
python experiments/e0_errata_checks.py          # errata for the base paper
python experiments/e1_gaussian_law.py           # variance-inflation law, exact Gaussian model
E2_THREADS=$T python experiments/e2_ideal_denoiser.py   # exact nonlinear denoiser

# -- 2. models --------------------------------------------------------------------------------
# A: standard rectified-flow training
OMP_NUM_THREADS=$T python experiments/train.py --steps 12000 --batch 64 --seed 0 --snapshots 3000 6000 \
    --out checkpoints/dit.pt --log results/train_log.json
# B: A + sufficiency-regularised fine-tune
OMP_NUM_THREADS=$T python experiments/train.py --init checkpoints/dit.pt --consistency 50 \
    --steps 3000 --batch 64 --lr 2e-4 --seed 1 \
    --out checkpoints/dit_regularised.pt --log results/train_log_regularised.json
# same with a gentler regulariser
OMP_NUM_THREADS=$T python experiments/train.py --init checkpoints/dit.pt --consistency 5 \
    --steps 3000 --batch 64 --lr 2e-4 --seed 1 \
    --out checkpoints/dit_regularised_w5.pt --log results/train_log_regularised_w5.json
# control: A + the same fine-tune without the regulariser ("is it just more training?")
OMP_NUM_THREADS=$T python experiments/train.py --init checkpoints/dit.pt \
    --steps 3000 --batch 64 --lr 2e-4 --seed 1 \
    --out checkpoints/dit_control.pt --log results/train_log_control.json

# -- 3. evaluation ----------------------------------------------------------------------------
python experiments/e3_trained_model.py --threads $T --ckpt checkpoints/dit.pt \
    --out results/e3_trained_model.json --save_examples results/e3_examples.pt
python experiments/e3_trained_model.py --threads $T --ckpt checkpoints/dit_regularised.pt \
    --out results/e3_regularised_model.json --save_examples results/e3_examples_regularised.pt
python experiments/e3_trained_model.py --lite --threads $T --ckpt checkpoints/dit_regularised_w5.pt \
    --out results/e3_regularised_w5_model.json --save_examples ""
python experiments/e6_sufficiency.py --threads $T --ckpts \
    "standard, 3k steps=checkpoints/dit_step3000.pt" "standard, 6k steps=checkpoints/dit_step6000.pt" \
    "standard, 12k steps=checkpoints/dit.pt" "+3k plain=checkpoints/dit_control.pt" \
    "+3k lambda=5=checkpoints/dit_regularised_w5.pt" "+3k lambda=50=checkpoints/dit_regularised.pt"
python experiments/e4_distribution_test.py --threads $T --ckpt checkpoints/dit.pt \
    --calib results/e3_trained_model.json --out results/e4_distribution_test.json
python experiments/e4_distribution_test.py --threads $T --ckpt checkpoints/dit_regularised.pt \
    --calib results/e3_regularised_model.json --out results/e4_distribution_test_regularised.json
python experiments/e5_wallclock.py --threads $T

# -- 4. paper ---------------------------------------------------------------------------------
python experiments/make_figures.py
python experiments/make_paper_inputs.py
python experiments/make_readme.py
(cd paper && pdflatex -interaction=nonstopmode paper.tex >/dev/null; bibtex paper >/dev/null; \
  pdflatex -interaction=nonstopmode paper.tex >/dev/null; pdflatex -interaction=nonstopmode paper.tex >/dev/null)
echo "done: paper/paper.pdf"
