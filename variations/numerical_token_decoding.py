import numpy as np
import torch


class NumericalTokenDecoder:
    def __init__(self, interpret: str = "uint", bitwidth: int = 16):
        self.interpret = interpret
        self.bitwidth = bitwidth
        self._validate_config()

    def _validate_config(self) -> None:
        valid_modes = {"uint", "sint", "fp16_bits", "bf16_bits"}
        if self.interpret not in valid_modes:
            raise ValueError(
                f"Unsupported numerical interpretation: {self.interpret}. "
                f"Expected one of {sorted(valid_modes)}."
            )
        if self.interpret == "sint" and self.bitwidth <= 0:
            raise ValueError("numerical_interpret_bitwidth must be positive for signed decoding.")
        if self.interpret == "sint" and self.bitwidth > 63:
            raise ValueError("numerical_interpret_bitwidth must be <= 63 for signed decoding.")

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.decode(tokens)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.interpret == "uint":
            return tokens
        if self.interpret == "sint":
            return self._decode_signed(tokens)
        if self.interpret == "fp16_bits":
            return self._decode_fp16_bits(tokens)
        if self.interpret == "bf16_bits":
            return self._decode_bf16_bits(tokens)
        raise ValueError(f"Unhandled numerical interpretation: {self.interpret}")

    def _decode_signed(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(torch.int64)
        mask = (1 << self.bitwidth) - 1
        unsigned = torch.bitwise_and(tokens, mask)
        sign_bit = 1 << (self.bitwidth - 1)
        signed = torch.where(unsigned >= sign_bit, unsigned - (1 << self.bitwidth), unsigned)
        return signed

    def _decode_fp16_bits(self, tokens: torch.Tensor) -> torch.Tensor:
        u16 = torch.bitwise_and(tokens.to(torch.int64), 0xFFFF).to(torch.uint16)
        bitcast = getattr(torch, "bitcast", None)
        if bitcast is not None:
            return bitcast(u16, torch.float16)
        try:
            return u16.view(torch.float16)
        except (TypeError, RuntimeError):
            array = u16.detach().cpu().numpy().view(np.float16)
            return torch.from_numpy(array).to(device=u16.device)

    def _decode_bf16_bits(self, tokens: torch.Tensor) -> torch.Tensor:
        u16 = torch.bitwise_and(tokens.to(torch.int64), 0xFFFF).to(torch.uint16)
        bitcast = getattr(torch, "bitcast", None)
        if bitcast is not None:
            return bitcast(u16, torch.bfloat16)
        try:
            return u16.view(torch.bfloat16)
        except (TypeError, RuntimeError):
            u32 = (u16.detach().cpu().numpy().astype(np.uint32) << 16)
            array = u32.view(np.float32)
            return torch.from_numpy(array).to(device=u16.device).to(torch.bfloat16)
