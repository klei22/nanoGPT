package com.nanogpt.tokenizer

/** Converts between app text and model token IDs. */
interface Tokenizer {
    fun encode(text: String): IntArray
    fun decode(tokens: IntArray): String
}
