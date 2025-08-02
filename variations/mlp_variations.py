# variations/mlp_variations.py

import torch
import torch.nn as nn

from variations.activation_variations import activation_dictionary
from variations.linear_variations import linear_dictionary
from quantization.quantize import fake_quantize_act
from quantization.quant_utils import set_variant, create_activation_buffers

class OriginalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        self.mlp_down_projs = config.mlp_down_projs

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        # Add learnable or fixed offsets for the activation function
        if config.learn_mlp_x_offset:
            self.activation_x_offset = nn.Parameter(torch.tensor(config.mlp_x_offset))
        else:
            self.register_buffer("activation_x_offset", torch.tensor(config.mlp_x_offset))

        if config.learn_mlp_y_offset:
            self.activation_y_offset = nn.Parameter(torch.tensor(config.mlp_y_offset))
        else:
            self.register_buffer("activation_y_offset", torch.tensor(config.mlp_y_offset))

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        mlp_expansion_size = None
        if config.mlp_size is not None:
            mlp_expansion_size = config.mlp_size
        else:
            mlp_expansion_size = config.mlp_expansion_factor * config.n_embd

        # Determine bias settings - use specific up/down if set, otherwise use global bias
        use_up_bias = config.mlp_up_bias if config.mlp_up_bias is not None else config.bias
        use_down_bias = config.mlp_down_bias if config.mlp_down_bias is not None else config.bias

        # Instantiate Linear Layers with configurable bias
        self.c_fc = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=use_up_bias
        )

        # Fused down projection
        self.c_proj = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd * self.mlp_down_projs,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=use_down_bias,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        x = self.c_fc(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_input", x, num_bits, quant_method, iter_num)

        # Apply offsets to the activation function
        x = self.activation_variant(x - self.activation_x_offset) - self.activation_y_offset

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_output", x, num_bits, quant_method, iter_num)


        # Apply fused down projection and sum the outputs
        x = self.c_proj(x)
        if self.mlp_down_projs > 1:
            batch_size, seq_len, _ = x.shape
            x = x.view(batch_size, seq_len, self.mlp_down_projs, -1)
            x = x.sum(dim=2)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x

class DualPathMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        # Dual path specific parameters
        if config.learn_mlp_x_offset:
            self.activation_x_offset = nn.Parameter(torch.tensor(config.mlp_x_offset))
        else:
            self.register_buffer("activation_x_offset", torch.tensor(config.mlp_x_offset))

        if config.learn_mlp_y_offset:
            self.activation_y_offset = nn.Parameter(torch.tensor(config.mlp_y_offset))
        else:
            self.register_buffer("activation_y_offset", torch.tensor(config.mlp_y_offset))

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        mlp_expansion_size = None
        if config.mlp_size is not None:
            mlp_expansion_size = config.mlp_size
        else:
            mlp_expansion_size = config.mlp_expansion_factor * config.n_embd

        # Instantiate Linear Layers with configurable bias
        self.c_fc = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=config.mlp_up_bias
        )

        # Two separate projection layers for each activation path
        self.c_proj1 = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=config.mlp_down_bias
        )

        self.c_proj2 = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=config.mlp_down_bias
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):
        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        # Common upscale projection
        x = self.c_fc(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_input", x, num_bits, quant_method, iter_num)

        # First activation path - shifted right
        x1 = self.activation_variant(x - self.activation_x_offset) - self.activation_y_offset
        x1 = self.c_proj1(x1)

        # Second activation path - shifted left and negated input
        x2 = -self.activation_variant(-(x + self.activation_x_offset)) - self.activation_y_offset
        x2 = self.c_proj2(x2)

        # Combine paths
        x = x1 + x2

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_output", x, num_bits, quant_method, iter_num)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)

        return x

class Swiglu(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler
        self.mlp_down_projs = config.mlp_down_projs

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        # Add learnable or fixed offsets for the activation function
        if config.learn_mlp_x_offset:
            self.activation_x_offset = nn.Parameter(torch.tensor(config.mlp_x_offset))
        else:
            self.register_buffer("activation_x_offset", torch.tensor(config.mlp_x_offset))

        if config.learn_mlp_y_offset:
            self.activation_y_offset = nn.Parameter(torch.tensor(config.mlp_y_offset))
        else:
            self.register_buffer("activation_y_offset", torch.tensor(config.mlp_y_offset))

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        mlp_expansion_size = None
        if config.mlp_size is not None:
            mlp_expansion_size = config.mlp_size
        else:
            mlp_expansion_size = config.mlp_expansion_factor * config.n_embd

        # Determine bias settings - use specific up/down if set, otherwise use global bias
        use_up_bias = config.mlp_up_bias if config.mlp_up_bias is not None else config.bias
        use_down_bias = config.mlp_down_bias if config.mlp_down_bias is not None else config.bias

        # Instantiate Linear Layers with configurable bias
        self.c_fc_in1 = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=use_up_bias
        )

        self.c_fc_in2 = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=use_up_bias
        )

        # Fused down projection
        self.c_fc_out = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd * self.mlp_down_projs,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=use_down_bias,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        x_in1 = self.c_fc_in1(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_input", x_in1, num_bits, quant_method, iter_num)

        x_in1 = self.activation_variant(x_in1 - self.activation_x_offset) - self.activation_y_offset

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_output", x_in1, num_bits, quant_method, iter_num)

        x_in2 = self.c_fc_in2(x)
        x_out = x_in1 * x_in2

        # Apply fused down projection and sum the outputs
        x = self.c_fc_out(x_out)
        if self.mlp_down_projs > 1:
            batch_size, seq_len, _ = x.shape
            x = x.view(batch_size, seq_len, self.mlp_down_projs, -1)
            x = x.sum(dim=2)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x

class KanMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.kan = linear_dictionary["kan"](config.n_embd, config.n_embd, config=config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):

        x = self.kan(x)
        x = self.dropout(x)

        return x

class MLP_Identity(nn.Module):
    def __init__(self, config):
        super(Identity, self).__init__()

    def forward(self, x, iter_num=None):
        return x

mlp_dictionary = {
    "mlp": OriginalMLP,
    "swiglu": Swiglu,
    "identity": MLP_Identity,
    "kan": KanMLP,
    "dual_path": DualPathMLP
    }

def get_mlp_instance(config):
    mlp_type = config.mlp_variant
    mlp_class = mlp_dictionary.get(mlp_type)
    if mlp_class is None:
        raise ValueError(f"Unsupported MLP variant: {mlp_type}")
    return mlp_class(config)

