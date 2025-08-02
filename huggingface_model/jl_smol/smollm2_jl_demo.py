import argparse
import math
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
import torch


def run_pipeline_demo():
    """Quick demo that the model can be loaded via HF pipeline."""
    pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct")
    messages = [{"role": "user", "content": "Who are you?"}]
    _ = pipe(messages)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
def _jl_project_tensor(tensor: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Project ``tensor`` using ``proj`` along matching dimensions."""
    in_dim = proj.shape[1]
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim >= 1 and tensor.shape[-1] == in_dim:
        tensor = tensor @ proj.t()
    if tensor.ndim > 1 and tensor.shape[0] == in_dim:
        tensor = proj @ tensor
    elif tensor.ndim == 1 and tensor.shape[0] == in_dim:
        tensor = (proj @ tensor.unsqueeze(-1)).squeeze(-1)
    return tensor


def apply_jl_transform(
    model: AutoModelForCausalLM,
    out_dim: int,
    std: float = 1.0,
    seed: int = 1337,
) -> AutoModelForCausalLM:
    """Apply a Gaussian JL transform to all model weights.

    If ``out_dim`` equals the current embedding dimension, the transform acts as
    a random rotation.  Otherwise the model is projected down to ``out_dim`` and
    a new model with reduced hidden size is returned.
    """

    hidden_size = model.config.hidden_size
    g = torch.Generator().manual_seed(seed)

    def gaussian_matrix(o, i):
        mat = torch.empty(o, i)
        mat.normal_(mean=0.0, std=std, generator=g)
        mat /= math.sqrt(o)
        return mat

    if out_dim == hidden_size:
        proj = gaussian_matrix(hidden_size, hidden_size)
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            if not torch.is_floating_point(tensor):
                continue
            tensor = _jl_project_tensor(tensor, proj)
            state_dict[name] = tensor
        model.load_state_dict(state_dict)
        return model

    # --- dimension reduction path ---
    print("model.config.num_attention_heads")
    print(model.config.num_attention_heads)
    if out_dim % model.config.num_attention_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_attention_heads")

    n_head = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    old_head_dim = hidden_size // n_head
    old_kv_dim = old_head_dim * n_kv
    new_head_dim = out_dim // n_head
    new_kv_dim = new_head_dim * n_kv
    old_mlp = model.config.intermediate_size
    new_mlp = int(old_mlp * out_dim / hidden_size)

    proj = gaussian_matrix(out_dim, hidden_size)
    proj_kv = gaussian_matrix(new_kv_dim, old_kv_dim)
    proj_mlp = gaussian_matrix(new_mlp, old_mlp)

    state_dict = model.state_dict()
    new_state = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            new_state[name] = tensor
            continue
        t = _jl_project_tensor(tensor, proj)
        if name.endswith(("k_proj.weight", "v_proj.weight")):
            t = _jl_project_tensor(t, proj_kv)
        if name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
            t = _jl_project_tensor(t, proj_mlp)
        if name.endswith("mlp.down_proj.weight"):
            t = t @ proj_mlp.t()
        new_state[name] = t

    from copy import deepcopy

    new_config = deepcopy(model.config)
    new_config.hidden_size = out_dim
    new_config.intermediate_size = new_mlp
    new_config.head_dim = new_head_dim
    new_model = model.__class__(new_config)
    new_model.load_state_dict(new_state, strict=False)
    new_model.tie_weights()
    return new_model


def main(args):
    run_pipeline_demo()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    def tok(examples):
        enc = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])
    train_ds = tokenized["train"]
    eval_ds = tokenized["validation"]

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    # training_args = TrainingArguments(
    #     output_dir="pretrain_smollm2",
    #     per_device_train_batch_size=2,
    #     per_device_eval_batch_size=2,
    #     max_steps=args.max_steps,
    #     logging_steps=args.log_interval,
    #     eval_strategy="steps",
    #     eval_steps=args.eval_interval,
    #     save_strategy="no",
    #     report_to=[],
    # )

    # trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds)
    # trainer.train()

    old_dim = model.config.hidden_size
    model   = apply_jl_transform(model, out_dim=args.hidden_dim, std=1.0)
    print(f"[JL] hidden_size: {old_dim} → {model.config.hidden_size}")

    finetune_args = TrainingArguments(
        output_dir="finetune_smollm2",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=args.max_steps,
        logging_steps=args.log_interval,
        eval_strategy="steps",
        eval_steps=args.eval_interval,
        save_strategy="no",
        report_to=[],
    )

    ft_trainer = Trainer(model=model, args=finetune_args, train_dataset=train_ds, eval_dataset=eval_ds)
    ft_trainer.train()

    train_steps = [x["step"] for x in ft_trainer.state.log_history if "loss" in x]
    train_losses = [x["loss"] for x in ft_trainer.state.log_history if "loss" in x]
    eval_steps = [x["step"] for x in ft_trainer.state.log_history if "eval_loss" in x]
    eval_losses = [x["eval_loss"] for x in ft_trainer.state.log_history if "eval_loss" in x]

    plt.plot(train_steps, train_losses, label="train")
    plt.plot(eval_steps, eval_losses, label="validation")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("finetune_loss.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=500,
        help="hidden dimension after JL projection, original is 576",
    )
    main(parser.parse_args())
