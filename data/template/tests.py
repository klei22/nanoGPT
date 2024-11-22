# tests.py

import unittest
import os
from tokenizers import (
    NumericRangeTokenizer,
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    ReplaceTokenizer,
    LinesTokenizer,
    CharTokenizer,
)
from argparse import Namespace


class TestTokenizers(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_text = "Hello, world!\nThis is a test."
        self.numeric_data = "123\n456\n789"
        self.tokens_file = "tokens.txt"

        # Create a tokens file for custom tokenizers
        with open(self.tokens_file, 'w') as f:
            f.write("Hello\nworld\nThis\nis\na\ntest\n")

    def tearDown(self):
        # Clean up tokens file
        if os.path.exists(self.tokens_file):
            os.remove(self.tokens_file)
        # Remove temporary files created by SentencePiece
        for fname in ["spm_input.txt", "trained_spm_model.model", "trained_spm_model.vocab"]:
            if os.path.exists(fname):
                os.remove(fname)
        if os.path.exists("meta.pkl"):
            os.remove("meta.pkl")
        if os.path.exists("remaining.txt"):
            os.remove("remaining.txt")

    def test_numeric_range_tokenizer(self):
        args = Namespace(min_token=100, max_token=1000)
        tokenizer = NumericRangeTokenizer(args)
        ids = tokenizer.tokenize(self.numeric_data)
        detokenized = tokenizer.detokenize(ids)
        self.assertEqual(self.numeric_data.strip(), detokenized)
        print("NumericRangeTokenizer test passed.")

    def test_sentencepiece_tokenizer(self):
        args = Namespace(
            vocab_size=30,
            spm_model_file=None,
            spm_vocab_file=None,  # Add this line
            skip_tokenization=False,
            byte_fallback=True
        )
        # Simulate training data
        with open("spm_input.txt", "w") as f:
            f.write(self.sample_text)
        tokenizer = SentencePieceTokenizer(args, input_files="spm_input.txt")
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)
        self.assertEqual(self.sample_text, detokenized)
        print("SentencePieceTokenizer test passed.")


    def test_tiktoken_tokenizer(self):
        args = Namespace(tiktoken_encoding='gpt2')
        tokenizer = TiktokenTokenizer(args)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)
        self.assertEqual(self.sample_text, detokenized)
        print("TiktokenTokenizer test passed.")

    # TODO implement tests for remaining tokenizations
    #def test_custom_tokenizer(self):
    #    pass

    #def test_replace_tokenizer(self):
    #    pass

    #def test_lines_tokenizer(self):
    #    pass

    def test_char_tokenizer(self):
        args = Namespace(reuse_chars=False)
        tokenizer = CharTokenizer(args, self.sample_text, None)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)
        self.assertEqual(self.sample_text, detokenized)
        print("CharTokenizer test passed.")


if __name__ == '__main__':
    unittest.main(verbosity=2)

