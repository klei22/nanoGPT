"""
Modular task benchmark system for nanoGPT.

Benchmarks are defined by JSON config files with the structure:
{
    "name": "counting",
    "description": "...",
    "start_tokens": ["A B C A|"],        # prompt(s) to feed the model
    "targets": ["2 1 1 2"],              # expected output(s)
    "grader": "benchmarks/grade_counting.py",  # script to grade
    "grader_args": "--T 10 --L 8",       # extra args for grader
    "max_new_tokens": 20,                # how many tokens to generate
    "temperature": 0.0,                  # greedy by default
    "num_eval_examples": 100             # how many examples to evaluate
}

The benchmark runner:
1. Loads the config
2. For each start_token, generates model output
3. Calls the grader script with: --generated <text> --target <text> [grader_args]
4. Collects scores and returns a summary dict

Can also generate start_tokens dynamically from a val_examples file.

Usage from train.py:
    from benchmarks.task_benchmarks import run_task_benchmarks
    results = run_task_benchmarks(model, encode, decode, device, config_paths, args)
"""

import json
import os
import subprocess
import sys
import tempfile

import torch


def load_benchmark_config(config_path):
    """Load a benchmark JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_from_prompt(model, encode, decode, device, prompt_str,
                         max_new_tokens=50, temperature=0.0, top_k=1,
                         block_size=1024):
    """Generate text from a prompt string using the model.

    Returns the generated text (excluding the prompt).
    """
    prompt_ids = encode(prompt_str)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    generated_ids = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                # Greedy
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, idx_next], dim=1)
            token_id = idx_next[0, 0].item()
            generated_ids.append(token_id)

            # Stop at newline for line-oriented tasks
            decoded_char = decode([token_id])
            if decoded_char == '\n':
                break

    return decode(generated_ids)


def load_val_examples(val_examples_path, num_examples=100, seed=42):
    """Load validation examples and split into (start_token, target) pairs.

    Expects lines like: 'A B D D B B|1 3 2 2 3 3'
    """
    import random
    rng = random.Random(seed)

    with open(val_examples_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) > num_examples:
        lines = rng.sample(lines, num_examples)

    start_tokens = []
    targets = []
    for line in lines:
        if '|' in line:
            parts = line.split('|', 1)
            start_tokens.append(parts[0] + '|')
            targets.append(parts[1])
        else:
            start_tokens.append(line)
            targets.append('')

    return start_tokens, targets


def run_grader(grader_script, grader_args, generated_texts, target_texts):
    """Run a grader script and return the score.

    The grader is called with:
        python <grader_script> <grader_args> --generated-file <tmp> --target-file <tmp>

    The grader should print a JSON dict to stdout with at least a 'score' key.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as gf:
        for text in generated_texts:
            gf.write(text.rstrip('\n') + '\n')
        gen_path = gf.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
        for text in target_texts:
            tf.write(text.rstrip('\n') + '\n')
        tgt_path = tf.name

    try:
        cmd = [
            sys.executable, grader_script,
            '--generated-file', gen_path,
            '--target-file', tgt_path,
        ]
        if grader_args:
            cmd.extend(grader_args.split())

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"Grader error ({grader_script}): {result.stderr}")
            return {'score': 0.0, 'error': result.stderr}

        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            # Try to extract just a number
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                try:
                    return {'score': float(line)}
                except ValueError:
                    continue
            return {'score': 0.0, 'error': f'Could not parse grader output: {result.stdout}'}
    finally:
        os.unlink(gen_path)
        os.unlink(tgt_path)


def run_single_benchmark(model, encode, decode, device, config,
                         config_dir=None, block_size=1024):
    """Run a single benchmark from its config dict.

    Returns a dict with 'name', 'score', and any extra grader outputs.
    """
    name = config.get('name', 'unknown')
    max_new_tokens = config.get('max_new_tokens', 50)
    temperature = config.get('temperature', 0.0)
    top_k = config.get('top_k', 1)
    num_examples = config.get('num_eval_examples', 100)
    grader = config.get('grader', '')
    grader_args = config.get('grader_args', '')

    # Resolve paths relative to config directory
    if config_dir and grader and not os.path.isabs(grader):
        # Grader paths are relative to repo root
        pass  # keep as-is, will be run from repo root

    # Get start_tokens and targets
    start_tokens = config.get('start_tokens', [])
    targets = config.get('targets', [])

    # If val_examples file is specified, load from there
    val_examples_file = config.get('val_examples_file', '')
    if val_examples_file and not start_tokens:
        if config_dir and not os.path.isabs(val_examples_file):
            val_examples_file = os.path.join(config_dir, val_examples_file)
        if os.path.exists(val_examples_file):
            start_tokens, targets = load_val_examples(
                val_examples_file, num_examples
            )

    if not start_tokens:
        return {'name': name, 'score': 0.0, 'error': 'No start_tokens or val_examples_file'}

    # Generate model outputs
    generated_texts = []
    model.eval()
    for prompt in start_tokens:
        output = generate_from_prompt(
            model, encode, decode, device, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            block_size=block_size,
        )
        generated_texts.append(output)

    # Run grader
    if grader and os.path.exists(grader):
        result = run_grader(grader, grader_args, generated_texts, targets)
    else:
        # Fallback: exact match scoring
        correct = sum(
            1 for g, t in zip(generated_texts, targets)
            if g.strip() == t.strip()
        )
        result = {
            'score': correct / len(generated_texts) if generated_texts else 0.0,
            'correct': correct,
            'total': len(generated_texts),
        }

    result['name'] = name
    return result


def run_task_benchmarks(model, encode, decode, device, config_paths,
                        block_size=1024, writer=None, iter_num=0):
    """Run all task benchmarks specified by config file paths.

    Args:
        model: The GPT model (will be set to eval mode)
        encode: Function to encode text to token ids
        decode: Function to decode token ids to text
        device: Device string
        config_paths: List of paths to benchmark JSON configs
        block_size: Model context window size
        writer: Optional tensorboard writer
        iter_num: Current training iteration (for logging)

    Returns:
        Dict mapping benchmark name -> result dict
    """
    results = {}
    model.eval()

    for config_path in config_paths:
        if not os.path.exists(config_path):
            print(f"  Benchmark config not found: {config_path}")
            continue

        config = load_benchmark_config(config_path)
        config_dir = os.path.dirname(os.path.abspath(config_path))
        result = run_single_benchmark(
            model, encode, decode, device, config,
            config_dir=config_dir, block_size=block_size,
        )

        name = result.get('name', 'unknown')
        score = result.get('score', 0.0)
        results[name] = result

        print(f"  Benchmark [{name}]: score={score:.4f}")
        for k, v in result.items():
            if k not in ('name', 'score', 'error') and isinstance(v, (int, float)):
                print(f"    {k}: {v}")

        # Log to tensorboard
        if writer is not None:
            writer.add_scalar(f"benchmark/{name}/score", score, iter_num)
            for k, v in result.items():
                if k not in ('name', 'error') and isinstance(v, (int, float)):
                    writer.add_scalar(f"benchmark/{name}/{k}", v, iter_num)

    return results
