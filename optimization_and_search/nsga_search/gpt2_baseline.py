import random, math, pickle, logging
from search_space import HeteroSearchSpace, Individual
from hw_exp import eval_individual
from hardware_search import evaluate_individual_on_hardware

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')


# define a gpt2 baseline model individual
def gpt2_baseline_individual_infi() -> Individual:
    global_spec = {
        "n_embd": 768,
        "block_size": 1024,
        "use_concat_heads": 0,
        "layer_mask": [1] * 12,
    }

    layer_spec = []
    for _ in range(12):
        layer = {
            "attention_variant": "infinity",
            "n_head": 12,
            "n_kv_group": 12,
            "n_qk_head_dim": 64,
            "n_v_head_dim": 64,
            "mlp_size": 3072,
            "n_cproj": 1,
        }
        layer_spec.append(layer)

    individual = Individual(globals=global_spec, layers=layer_spec)
    return individual

def smollm2_baseline_individual_infi() -> Individual:
    global_spec = {
        "n_embd": 576,
        "block_size": 512,
        "use_concat_heads": 1,
        "layer_mask": [1] * 30,
    }

    layer_spec = []
    for _ in range(30):
        layer = {
            "attention_variant": "infinity",
            "n_head": 9,
            "n_kv_group": 3,
            "n_qk_head_dim": 64,
            "n_v_head_dim": 64,
            "mlp_size": 1536,
            "n_cproj": 1,
        }
        layer_spec.append(layer)

    individual = Individual(globals=global_spec, layers=layer_spec)
    return individual

def gpt2_baseline_individual_mha() -> Individual:
    global_spec = {
        "n_embd": 768,
        "block_size": 1024,
        "use_concat_heads": 0,
        "layer_mask": [1] * 12,
    }

    layer_spec = []
    for _ in range(12):
        layer = {
            "attention_variant": "causal",
            "n_head": 12,
            "n_kv_group": 12,
            "n_qk_head_dim": 64,
            "n_v_head_dim": 64,
            "mlp_size": 3072,
            "n_cproj": 1,
        }
        layer_spec.append(layer)

    individual = Individual(globals=global_spec, layers=layer_spec)
    return individual


def main():
    # inf_infi = gpt2_baseline_individual_infi()
    inf_infi = smollm2_baseline_individual_infi()
    work_dir = "./hw_eval/runs/"
    individual_stats = eval_individual(inf_infi, work_dir=work_dir)
    # print("GPT2-Infinity Baseline Individual Hardware Stats on Timeloop:\n", individual_stats)
    print("SMOLLM2-Infinity Baseline Individual Hardware Stats on Timeloop:\n", individual_stats)

    # infi_mha = gpt2_baseline_individual_mha()
    # onasic_stats = evaluate_individual_on_hardware(infi_mha)
    # print("GPT2-MHA Baseline Individual Hardware Stats on Onasic:\n", onasic_stats)


if __name__ == "__main__":
    main()



