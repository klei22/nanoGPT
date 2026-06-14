package com.nanogpt.tokenizer

import org.json.JSONObject

/**
 * Character-level tokenizer backed by android_app/app/src/main/assets/tokenizer.json.
 *
 * This intentionally implements only the simple stoi/itos JSON format exported by
 * android_export/export_tokenizer.py. GPT-2/tiktoken, SentencePiece, Hugging Face,
 * byte-fallback, and BPE tokenizers need tokenizer-specific Android implementations
 * and must not be routed through this class.
 */
class JsonCharTokenizer private constructor(
    private val stoi: Map<String, Int>,
    private val itos: Map<Int, String>,
    val vocabSize: Int,
) : Tokenizer {

    override fun encode(text: String): IntArray {
        val ids = ArrayList<Int>(text.length)
        var offset = 0
        while (offset < text.length) {
            val codePoint = text.codePointAt(offset)
            val token = String(Character.toChars(codePoint))
            val id = stoi[token]
                ?: throw IllegalArgumentException("Character is not in tokenizer vocabulary: $token")
            ids.add(id)
            offset += Character.charCount(codePoint)
        }
        return ids.toIntArray()
    }

    override fun decode(tokens: IntArray): String {
        val out = StringBuilder(tokens.size)
        for (token in tokens) {
            val piece = itos[token]
                ?: throw IllegalArgumentException("Token ID is not in tokenizer vocabulary: $token")
            out.append(piece)
        }
        return out.toString()
    }

    companion object {
        fun fromJson(json: String): JsonCharTokenizer {
            val root = JSONObject(json)
            val tokenizerType = root.getString("tokenizer_type")
            require(tokenizerType == "char" || tokenizerType == "character" || tokenizerType == "char_level") {
                "JsonCharTokenizer only supports char-level JSON, got tokenizer_type=$tokenizerType"
            }

            val stoiJson = root.getJSONObject("stoi")
            val itosJson = root.getJSONObject("itos")
            val stoi = mutableMapOf<String, Int>()
            val stoiKeys = stoiJson.keys()
            while (stoiKeys.hasNext()) {
                val token = stoiKeys.next()
                require(!token.startsWith("byte:")) {
                    "byte-fallback tokens require a tokenizer-specific Android implementation"
                }
                stoi[token] = stoiJson.getInt(token)
            }

            val itos = mutableMapOf<Int, String>()
            val itosKeys = itosJson.keys()
            while (itosKeys.hasNext()) {
                val idKey = itosKeys.next()
                val token = itosJson.getString(idKey)
                require(!token.startsWith("byte:")) {
                    "byte-fallback tokens require a tokenizer-specific Android implementation"
                }
                itos[idKey.toInt()] = token
            }
            val vocabSize = root.getInt("vocab_size")
            require(stoi.size == vocabSize && itos.size == vocabSize) {
                "Tokenizer JSON size mismatch: stoi=${stoi.size}, itos=${itos.size}, vocab_size=$vocabSize"
            }
            return JsonCharTokenizer(stoi, itos, vocabSize)
        }
    }
}
