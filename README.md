# DDPM with Self-Attention in PyTorch

An educational implementation of an unconditional Denoising Diffusion Probabilistic Model (DDPM) for generating 64 x 64 RGB face images. The project implements the diffusion process and denoising network directly in PyTorch, including v-prediction, exponential moving average (EMA) weights, sinusoidal timestep embeddings, residual blocks, and bottleneck self-attention.

The model was trained on a 2,000-image subset of CelebA selected using facial attributes. This repository documents my hands-on exploration of the mathematical and implementation foundations behind diffusion models.

## Highlights

- Custom 14.1M-parameter U-Net implemented from scratch
- Sinusoidal timestep embeddings injected into residual blocks
- Self-attention at the 16 x 16 bottleneck resolution
- Linear noise schedule with 500 diffusion steps
- v-prediction training objective
- EMA model weights for more stable sampling
- Periodic checkpointing, sample generation, and loss visualization
- End-to-end CelebA filtering and preprocessing workflow

## Model overview

The denoising network receives a noisy image `x_t` and its diffusion timestep `t`, and predicts the velocity parameterization `v`.

```text
64 x 64 RGB image
       |
Initial convolution, 128 channels
       |
Residual block at 64 x 64
       |
Residual block at 32 x 32
       |
Residual block at 16 x 16
       |
Residual block and self-attention
       |
Skip-connected upsampling path
       |
Predicted v, 3 x 64 x 64
```

For a clean image `x_0`, sampled noise `epsilon`, and cumulative noise coefficient `alpha_bar_t`, the forward process constructs:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

Instead of predicting `epsilon` directly, the network is trained against:

```text
v = sqrt(alpha_bar_t) * epsilon
  - sqrt(1 - alpha_bar_t) * x_0
```

During sampling, the predicted velocity is converted back into an estimate of the noise before applying each reverse-diffusion step.

## Experiment configuration

| Setting | Value |
|---|---:|
| Dataset | CelebA subset |
| Training images | 2,000 |
| Image resolution | 64 x 64 RGB |
| Model parameters | 14.11M |
| Diffusion steps | 500 |
| Beta schedule | Linear, `1e-4` to `1e-2` |
| Training objective | Mean-squared error on v-prediction |
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Batch size | 8 |
| Epochs | 150 |
| EMA decay | `0.999` |

The recorded average training loss decreased from approximately `0.0524` in the first epoch to `0.0453` at epoch 149. Training loss alone is not a measure of perceptual sample quality; generated sample grids should be inspected alongside quantitative metrics in future experiments.

## Dataset preparation

The notebook uses the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). For the recorded experiment, images matching the following attributes were selected:

- `Young = 1`
- `Smiling = 1`
- `Male = -1`
- `No_Beard = 1`

A reproducible random sample of 2,000 matching images was then used for training. Images are resized and center-cropped to 64 x 64 pixels and normalized to `[-1, 1]`.

CelebA is not included in this repository. Download and use it in accordance with its official terms.

## Running the notebook

### Requirements

- Python 3.10 or newer
- PyTorch
- torchvision
- pandas
- Pillow
- matplotlib
- tqdm
- Jupyter Notebook or JupyterLab
- A CUDA-capable GPU is strongly recommended

Install the main dependencies with:

```bash
pip install torch torchvision pandas pillow matplotlib tqdm jupyter kagglehub
```

### Usage

1. Download CelebA from its official source or through Kaggle.
2. Update the dataset and checkpoint paths in the notebook.
3. Run the dataset-filtering and preprocessing cells.
4. Verify that the transformed batch has shape `[B, 3, 64, 64]` and values in `[-1, 1]`.
5. Run the architecture, training, and sampling cells in order.
6. Load the final EMA checkpoint to generate a new sample grid.

The current notebook was developed in a hosted GPU workspace and therefore contains absolute `/workspace/...` paths. Replace these paths before running it in another environment.

## Current limitations

- Unconditional generation only; no class, text, or image conditioning
- Small and deliberately filtered training subset
- Fixed 64 x 64 resolution
- Linear beta schedule and ancestral sampling only
- Sampling requires all 500 reverse steps
- No held-out FID, KID, or precision-recall evaluation
- Paths and experiment parameters are not yet controlled by one configuration
- Dataset preparation and model experiments are mixed in a single notebook
- Reproducibility is incomplete because all random-number generators are not seeded

## Planned improvements

- Consolidate configuration and remove duplicated exploratory cells
- Move training and sampling into command-line scripts
- Add deterministic seeding and environment documentation
- Export representative samples and the loss curve to `assets/`
- Add FID or KID evaluation on a held-out set
- Compare self-attention against an otherwise identical U-Net baseline
- Add faster DDIM sampling
- Explore conditional generation using text or visual embeddings
- Extend the work toward diffusion policies or multimodal embodied prediction

## Motivation

I built this project to understand diffusion models beyond using a high-level generation library. Implementing the forward process, timestep-conditioned denoiser, v-prediction objective, EMA updates, and reverse sampler made the relationship between diffusion theory and practical training behavior concrete.

The project also forms part of my preparation for research in generative computer vision, multimodal learning, and embodied AI.

## References

- Ho, J., Jain, A., and Abbeel, P. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), NeurIPS 2020.
- Nichol, A. Q., and Dhariwal, P. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), ICML 2021.
- Salimans, T., and Ho, J. [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512), ICLR 2022.
