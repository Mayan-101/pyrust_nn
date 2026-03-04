# pyrust_nn

[![Rust](https://img.shields.io/badge/Rust-1.80+-93450F?logo=rust)](https://www.rust-lang.org/) [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)

**pyrust_nn** is a hybrid Python-Rust framework for efficient fine-tuning, quantization, and deployment of large language models (LLMs). It leverages Python's rich ecosystem (Transformers, PEFT, etc.) for core ML tasks while using Rust for robust pipeline orchestration, logging, and cross-language integration via PyO3. The project supports both full fine-tuning and LoRA (Low-Rank Adaptation) on models like Qwen/Qwen3-1.7B, with built-in GGUF conversion for lightweight inference (e.g., with llama.cpp).

This setup is ideal for researchers or developers wanting a reproducible, high-performance workflow for LLM customization—running entirely on GPU (CUDA required).

## Features

- **Hybrid Architecture**: Python for ML-heavy operations (fine-tuning, inference); Rust for pipeline control, error handling, and session-based logging.
- **Fine-Tuning Modes**:
  - Full fine-tuning with gradient checkpointing and bfloat16 precision.
  - LoRA fine-tuning with configurable rank, alpha, and dropout.
- **Quantization**: 4-bit, 8-bit, or 16-bit quantization using BitsAndBytes.
- **GGUF Export**: Convert models/adapters to GGUF format for efficient CPU/GPU inference.
- **Inference**: Chat-style generation with thinking traces (Qwen-specific).
- **Session Management**: Per-run logging and artifact storage in `runs/<session_id>/`.
- **Extensibility**: NIF (Rustler) integration for Erlang/Elixir; easy parameter tuning via JSON.
- **Monitoring**: ETA callbacks and detailed logs for long-running training.

## Quick Start

### Prerequisites
- **Hardware**: NVIDIA GPU with CUDA 11.8+ (at least 8GB VRAM for Qwen-1.7B).
- **Software**:
  - Python 3.10+.
  - Rust 1.80+ (stable channel).
  - Git.

### Installation
1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd pyrust_nn
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build the Rust crate:
   ```bash
   cargo build --release
   ```

4. Prepare your dataset: Create `data.json` in the root (Qwen chat format example):
   ```json
   [
     {
       "messages": [
         {"role": "user", "content": "What is AI?"},
         {"role": "assistant", "content": "AI is artificial intelligence..."}
       ]
     }
   ]
   ```

### Running the Pipeline
Execute the full workflow (LoRA → GGUF, Full FT → GGUF, Inference) via Rust:
```bash
cargo run --release
```
- Outputs: Artifacts in `runs/RandomSession/` (logs, models, GGUF files).
- Customize session ID or params in `src/main.rs`.

For isolated steps, use Python scripts directly (e.g., `python finetuning_lora.py`).

## Usage

### Pipeline Overview
The Rust binary (`src/main.rs`) orchestrates a multi-step workflow:
1. **LoRA Fine-Tuning**: Train a lightweight adapter on your dataset.
2. **LoRA to GGUF**: Export adapter for fast inference.
3. **Full Fine-Tuning**: Train the entire model (resource-intensive).
4. **Full Model to GGUF**: Export the fine-tuned model.
5. **Inference Test**: Generate responses from the fine-tuned model.

Logs and summaries are saved per step (e.g., `runs/<session_id>/finetune_lora/summary.json`).

### Python Scripts
- **`finetune_full.py`**: Full fine-tuning. Run: `python finetune_full.py`.
- **`finetuning_lora.py`**: LoRA fine-tuning. Supports resuming from checkpoints.
- **`inference.py`**: Standalone inference with chat templates.
- **`quant.py`**: Quantize a model (e.g., to 4-bit for reduced memory).

### Rust API
Expose functions via `lib.rs` for embedding in other Rust/Elixir apps:
- `finetune_lora(...)` → Returns adapter path.
- `run_inference(...)` → Returns generated content.
- NIFs (e.g., `finetune_lora_nif`) for Erlang integration.

### Configuration
Edit `params.json` for defaults:
```json
{
  "model_name": "Qwen/Qwen3-1.7B",
  "num_epochs": 2,
  "batch_size": 1,
  "lora_rank": 8,
  "gguf_precision": "q8_0",
  ...
}
```
Pass overrides to functions (e.g., in `main.rs` or Python calls).

### GGUF Conversion
Requires `llama.cpp` tools (not included—install separately). Outputs like `model.gguf` or `lora_adapter.gguf` for deployment.

## Project Structure
```
pyrust_nn/
├── Cargo.toml          # Rust dependencies (PyO3, Rustler, etc.)
├── params.json         # Config defaults
├── requirements.txt    # Python deps (Transformers, PEFT, etc.)
├── finetune_full.py    # Full FT script
├── finetuning_lora.py  # LoRA FT script
├── inference.py        # Inference script
├── quant.py            # Quantization script
└── src/
    ├── lib.rs          # Core API + PyO3 bindings + NIFs
    └── main.rs         # Pipeline orchestrator
```
- `runs/`: Auto-generated (logs, models per session).

## Dependencies

### Python
See `requirements.txt` (key: `torch==2.7.0`, `transformers==4.53.3`, `peft==0.17.1`).

### Rust
See `Cargo.toml` (key: `pyo3==0.25.1`, `rustler==0.36.2` for NIFs).

## Troubleshooting
- **CUDA Errors**: Ensure `torch` sees your GPU (`python -c "import torch; print(torch.cuda.is_available())"`).
- **Memory Issues**: Reduce `batch_size` or enable `gradient_checkpointing`.
- **Rust-PyO3 Fails**: Run `cargo clean && cargo build` after Python env changes.
- **GGUF Errors**: Verify `llama.cpp` convert.py is in PATH (custom impl in pipeline).


---
