"""
Implements the evaluation metrics based on BLEU score

example:
    import sacrebleu

    translated_sentences = ['The dog had bit the man.', "It wasn't surprising.", 'The man had bitten the dog.']
    target_sentences = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    bleu_score = sacrebleu.corpus_bleu(translated_sentences, [target_sentences]).score
    print(f'Test BLEU: {bleu_score}')

"""

import numpy as np
from typing import List

import sacrebleu

def corpus_bleu(sys_sents: List[str],
                refs_sents: List[List[str]],
                smooth_method: str = 'exp',
                smooth_value: float = None,
                force: bool = True,
                lowercase: bool = False,
                tokenizer: str = '13a',
                use_effective_order: bool = False):

    return sacrebleu.corpus_bleu(sys_sents, refs_sents, smooth_method, smooth_value, force,
                                 lowercase=False, tokenize='none', use_effective_order=use_effective_order).score


def sentence_bleu(sys_sent: str,
                  ref_sents: List[str],
                  smooth_method: str = 'floor',
                  smooth_value: float = None,
                  lowercase: bool = False,
                  tokenizer: str = '13a',
                  use_effective_order: bool = True):

    return corpus_bleu([sys_sent], [[ref] for ref in ref_sents], smooth_method, smooth_value, force=True,
                       lowercase=lowercase, tokenizer=tokenizer, use_effective_order=use_effective_order)


def corpus_averaged_sentence_bleu(sys_sents: List[str],
                                  refs_sents: List[List[str]],
                                  smooth_method: str = 'floor',
                                  smooth_value: float = None,
                                  lowercase: bool = False,
                                  tokenizer: str = '13a',
                                  use_effective_order: bool = True):

    scores = []
    for sys_sent, *ref_sents in zip(sys_sents, *refs_sents):
        scores.append(sentence_bleu(sys_sent, ref_sents, smooth_method, smooth_value,
                                    lowercase=lowercase, tokenizer=tokenizer, use_effective_order=use_effective_order))
    return np.mean(scores)
