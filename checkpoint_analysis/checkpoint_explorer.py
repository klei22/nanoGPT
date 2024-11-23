import os
import sys
import argparse
import torch

# Add top level dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Model Parameter Explorer")
    parser.add_argument("ckpt_path", help="Path to the checkpoint file")
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
    return parser.parse_args()

def load_model(ckpt_path, device):
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint.get('model_args', None)
    if model_args is None:
        sys.exit("Model arguments not found in checkpoint.")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_parameter_tree(state_dict):
    tree = {}
    for full_key in state_dict.keys():
        parts = full_key.split('.')
        current_level = tree
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        # Last part is the parameter tensor
        current_level[parts[-1]] = state_dict[full_key]
    return tree

def explore_tree(tree, path=[]):
    while True:
        current_level = tree
        for part in path:
            current_level = current_level[part]

        if isinstance(current_level, dict):
            keys = list(current_level.keys())
            print("\nCurrent Path: " + '.'.join(path) if path else "root")
            print("Submodules/Parameters:")
            for idx, key in enumerate(keys):
                print(f"{idx}: {key}")
            print("b: Go back, q: Quit")
            choice = input("Enter the number of the submodule/parameter to explore (or 'b' to go back, 'q' to quit): ")
            if choice == 'b':
                if path:
                    path.pop()
                else:
                    print("Already at root.")
            elif choice == 'q':
                break
            elif choice.isdigit() and int(choice) < len(keys):
                path.append(keys[int(choice)])
            else:
                print("Invalid choice.")
        else:
            # It's a parameter tensor
            full_key = '.'.join(path)
            tensor = current_level
            tensor_str = str(tensor.detach().cpu().numpy())
            if len(tensor_str) > 1000:
                tensor_str = tensor_str[:1000] + '...'
            print(f"\nValue of {full_key}:")
            print(tensor_str)
            input("Press Enter to continue...")
            path.pop()

def main():
    args = parse_args()

    model = load_model(args.ckpt_path, args.device)
    state_dict = model.state_dict()

    parameter_tree = get_parameter_tree(state_dict)

    print("Model Parameter Explorer")
    print("Navigate through the parameters using numbers. Press 'b' to go back, 'q' to quit.")

    explore_tree(parameter_tree)

if __name__ == '__main__':
    main()

