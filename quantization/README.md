## Quantization Capabilities

Our repo provides custom quantization options for activations and weights quantization of attention and mlp in the transformer. 

## Example Usage

Below is an example training command to train a fully quantized model:

```bash
python3 train.py \
--quantize_linear_method="symmetric_quant" \
--activations_quant_method="symmetric_quant" \
--dtype="bfloat16" \
--quantization_warmup_iters=0 \
--quantize_attn_act \
--quantize_mlp_act \
--linear_variant_attn="quantized_linear" \
--linear_variant_mlp="quantized_linear" \
--store_activations
```

## Saving Quantized Values

The quantization/save_weights.py file can be used to save the quantized weights and activations (default to a .pkl file). 

--device: Device to load the checkpoint ('cpu', 'cuda'). Default: 'cuda'.
--out_dir: Directory of the checkpoint. Default: 'out'.
--file_name: Output file name. Default: 'quantized_weights'.
--file_type: Output file type ('pkl'). Default: 'pkl'.

Below is an example command to save a model's quantized values to a .pkl file:

```bash
python3 quantization/save_weights.py \
--out_dir="quantized_model" \
--file_name="quantized_data" \
--file_type="pkl"
```

## Visualizing Quantized Values

The quantization/visualize.py file can be used to visualize the weights and activations. This script loads the quantized weights and activations and generates histograms or matrix heatmaps for analysis.

Below is an example command to create a histogram for every quantized weight and activation of a model:

```bash
python3 quantization/visualize.py \
--file_name="quantized_data.pkl" \
--image_folder="quantized_images" \
--weight="all" \
--graph="histogram"
```

## Quantization Methods

- **symmetric_quant**: Quantizes values symmetrically around zero (zero point is 0).
- **affine_quant**: Uses an asymmetric transformation for quantization.
- **stochastic_quant**: Introduces randomness into the quantization process for stochastic rounding.
- **ternary_quant**: Symmetrically quantizes values to a 1.58 bit range of -1, 0, and 1

---

## Linear Variants

- **quantized_linear**: A linear layer where the weights are quantized.

### Configuration Options

- **Global Linear Variant for Attention Layers**:
  - `--linear_variant_attn`: Sets the linear variant for all attention layers.

- **Specific Linear Variants for Attention Sub-layers**:
  - `--linear_variant_q`: Linear variant for the query projection (`c_attn_q`).
  - `--linear_variant_k`: Linear variant for the key projection (`c_attn_k`).
  - `--linear_variant_v`: Linear variant for the value projection (`c_attn_v`).
  - `--linear_variant_attn_proj`: Linear variant for the output projection (`c_proj`) in attention.
  - **Note**: These options take precedence over `--linear_variant_attn` if specified.

- **Global Linear Variant for MLP Layers**:
  - `--linear_variant_mlp`: Sets the linear variant for all MLP layers.
    - **Default**: `linear`

- **Specific Linear Variants for MLP Sub-layers**:
  - `--linear_variant_mlp_up`: Linear variant for the up-projection (`c_fc`) in MLP.
  - `--linear_variant_mlp_down`: Linear variant for the down-projection (`c_proj`) in MLP.
  - **Note**: These options take precedence over `--linear_variant_mlp` if specified.

---

## Activation Quantization

### Attention Activations

- **Quantize All Attention Activations**:
  - `--quantize_attn_act`: If set, quantizes all input and output activations within attention layers.
    - **Default**: `False`

- **Granular Control Over Attention Activation Quantization**:

  - **Input to Attention Layer**:
    - `--quantize_attn_act_input`: Quantizes the input to the attention layer.

  - **Query and Key Inputs to QK Multiplication**:
    - `--quantize_attn_act_qk_mult_q_input`: Quantizes the query input.
    - `--quantize_attn_act_qk_mult_k_input`: Quantizes the key input.

  - **Softmax Input**:
    - `--quantize_attn_act_softmax_input`: Quantizes the input to the softmax function.

  - **Probability and Value Inputs to PV Multiplication**:
    - `--quantize_attn_act_pv_mult_p_input`: Quantizes the softmax output (probabilities).
    - `--quantize_attn_act_pv_mult_v_input`: Quantizes the value input.

  - **Outputs**:
    - `--quantize_attn_act_pv_mult_output`: Quantizes the output of the PV multiplication.
    - `--quantize_attn_act_output`: Quantizes the output of the attention layer.

- **Default Precision for Attention Activations**:
  - `--quantize_attn_act_bits`: Number of bits for quantizing attention activations.
    - **Default**: `8`

- **Granular Precision Overrides**:

  - For each of the granular options above, you can specify the number of bits:

    - `--quantize_attn_act_input_bits`
    - `--quantize_attn_act_qk_mult_q_input_bits`
    - `--quantize_attn_act_qk_mult_k_input_bits`
    - `--quantize_attn_act_softmax_input_bits`
    - `--quantize_attn_act_pv_mult_p_input_bits`
    - `--quantize_attn_act_pv_mult_v_input_bits`
    - `--quantize_attn_act_pv_mult_output_bits`
    - `--quantize_attn_act_output_bits`

  - **Note**: If not specified, these default to the value of `--quantize_attn_act_bits`.

### MLP Activations

- **Quantize All MLP Activations**:
  - `--quantize_mlp_act`: If set, quantizes all input and output activations within MLP layers.
    - **Default**: `False`

- **Granular Control Over MLP Activation Quantization**:

  - **Input to MLP Layer**:
    - `--quantize_mlp_act_input`: Quantizes the input to the MLP layer.

  - **Activation Function Input and Output**:
    - `--quantize_mlp_act_activation_input`: Quantizes the input to the activation function.
    - `--quantize_mlp_act_activation_output`: Quantizes the output of the activation function.

  - **Output of MLP Layer**:
    - `--quantize_mlp_act_output`: Quantizes the output of the MLP layer.

- **Default Precision for MLP Activations**:
  - `--quantize_mlp_act_bits`: Number of bits for quantizing MLP activations.
    - **Default**: `8`

- **Granular Precision Overrides**:

  - For each of the granular options above, you can specify the number of bits:

    - `--quantize_mlp_act_input_bits`
    - `--quantize_mlp_act_activation_input_bits`
    - `--quantize_mlp_act_activation_output_bits`
    - `--quantize_mlp_act_output_bits`

  - **Note**: If not specified, these default to the value of `--quantize_mlp_act_bits`.

- **Store Activations**:
  - `--store_activations`: If set, saves the activations as buffers and updates them during training.
    - **Default**: `False`

---

## Quantization Warmup Iterations

- **Warmup Iterations**:
  - `--quantization_warmup_iters`: Specifies the number of iterations to train using regular (non-quantized) linear layers before switching to quantized linear layers.
    - **Default**: `100`
    - **Purpose**: Allows the model to stabilize before introducing quantization, which can improve training convergence.

---

## SpinQuant Rotations

The `spin_quant.py` module implements the core ideas from the *SpinQuant* paper.
It learns a small rotation matrix that reduces the quantization error prior to
post training quantization. A typical usage looks like:

```python
import torch
from model import GPT
from quantization.spin_quant import SpinQuant

model = GPT.from_pretrained('gpt2')  # load your model
sq = SpinQuant(model, bits=4)
calib_data = [torch.randint(0, model.config.vocab_size, (1, 16)) for _ in range(8)]
# optimise rotation on calibration data
sq.optimize(calib_data, steps=100)
# apply rotation and quantize weights in-place
sq.apply()
```

The resulting model has its weights rotated and quantized. The `SpinQuant`
class can also be incorporated into training loops by calling `optimize` during
finetuning before applying the rotation.

The `demos/spinquant_ptq.py` script demonstrates an end-to-end workflow where a
checkpoint produced by `train.py` is loaded, optimized with `SpinQuant` and then
saved back to disk for inference. For QAT, call `SpinQuant.optimize` on a few
training batches periodically and invoke `SpinQuant.apply()` after training to
bake the rotation into the weights.
You can then run inference using the checkpoint produced by `spinquant_ptq.py` via:

```bash
python3 demos/spinquant_sample.py --ckpt spinquant_out/ckpt_spinquant.pt --prompt "Test" --max_new_tokens 20
```
Ensure that `meta.pkl` from the training run resides in `spinquant_out` so the
inference script can tokenize inputs correctly.
