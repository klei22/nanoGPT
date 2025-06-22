# ReaLLMASIC

![Project Logo](docs/images/logo_placeholder.png)

Welcome to **ReaLLMASIC**, an extension of the [nanoGPT](https://github.com/karpathy/nanoGPT) project. This repository focuses on hardware‑aware experimentation and highly modular model components. It is designed for researchers and practitioners who want to explore different GPT variations and measure their impact on power, performance, and area (PPA).

<Placeholder for intro video demonstrating the project>

## Key Features

- **Module Variations** – Swap in alternative attention, MLP, normalization, or softmax implementations from the `variations/` directory.
- **Flexible Tokenization** – Choose between tiktoken, SentencePiece, phoneme tokenizers, or your own custom tokenizer.
- **Diverse Dataset Support** – Train on datasets ranging from literature and mathematics to music, timeseries, and more.
- **Hardware Awareness** – Tools for quantization, Verilog modules under `HW/SA`, and utilities for mobile export with ExecuTorch.
- **Experiment Automation** – Run large parameter sweeps using `optimization_and_search/run_experiments.py` and view logs in `csv_logs/` and `logs/`.
- **Comprehensive Logging** – Timestamped output folders and optional TensorBoard/WandB integration.

![Feature Overview](docs/images/feature_overview_placeholder.png)

## Repository Structure

- `model.py` – Core GPT model that imports the desired variation modules.
- `train.py` – Main training script, using arguments defined in `train_args.py`.
- `variations/` – Collection of modules that implement different activations, softmaxes, norms, routers, and more.
- `optimization_and_search/` – Utilities for sweeping over configurations.
- `analysis/` – Scripts such as `checkpoint_explorer.py` to inspect checkpoints.
- `exutorch/` – Example scripts for exporting the model with PyTorch’s ExecuTorch.
- `quantization/` – Training and visualization tools for quantized models.

## Quick Start

1. **Install Dependencies** (GPU recommended):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install -r requirements_gpu.txt
```

2. **Prepare a Dataset and Train**:

```bash
bash data/shakespeare_char/get_dataset.sh
python3 train.py --compile --max_sample_tokens 100
```

3. **Sample From the Model**:

```bash
python3 sample.py --out_dir out
```

![Training Screenshot](docs/images/training_placeholder.png)

## Exploration

Use `run_experiments.py` to launch sweeps over multiple configurations.

```bash
python3 optimization_and_search/run_experiments.py -c explorations/config.json
```

Check progress and inspect the best validation losses:

```bash
python3 checkpoint_analysis/inspect_ckpts.py --directory ./out --sort loss
```

![Experiment Dashboard](docs/images/experiment_dashboard_placeholder.png)

## Contributing

We welcome contributions that add new model variations, datasets, or analysis tools. See [`documentation/Contributing_Features.md`](documentation/Contributing_Features.md) for guidelines and pull request tips.

## Acknowledgements

- Built on the excellent [nanoGPT](https://github.com/karpathy/nanoGPT) project
- Inspired by the [Zero To Hero](https://karpathy.ai/zero-to-hero.html) series

