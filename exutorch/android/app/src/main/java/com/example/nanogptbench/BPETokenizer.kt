package com.example.nanogptbench

import android.content.res.AssetManager
import org.json.JSONObject
import java.io.BufferedReader

/**
 * Pure-Kotlin GPT-2 Byte-Pair Encoding tokenizer.
 *
 * Loads two asset files:
 *   encoder.json  – { "token_string": id, … }   (vocab.json from HuggingFace)
 *   vocab.bpe     – BPE merge rules, one per line:  "Ġ t"  →  "Ġt"
 *
 * The implementation follows the original GPT-2 tokenizer exactly:
 *   1. Split text with the GPT-2 regex (apostrophes, letters, digits, etc.)
 *   2. Encode each byte of each word as a printable unicode char via the
 *      bytes_to_unicode() mapping.
 *   3. Apply BPE merges greedily (lowest rank first).
 *   4. Map resulting tokens to integer IDs.
 *
 * Decoding reverses step 4 → step 2 (unicode → bytes → UTF-8 string).
 */
class BPETokenizer(assets: AssetManager) {

    // Token string → integer ID
    private val encoder = mutableMapOf<String, Int>()
    // Integer ID → token string
    private val decoder = mutableMapOf<Int, String>()
    // BPE merge rules: pair → rank (lower rank = higher priority)
    private val bpeRanks = mutableMapOf<Pair<String, String>, Int>()
    // Byte value (0–255) → single unicode character
    private val byteEncoder = buildByteEncoder()
    // Inverse: single unicode character → byte value
    private val byteDecoder = byteEncoder.entries.associate { (k, v) -> v to k }

    // Memoisation for BPE merge sequences
    private val bpeCache = mutableMapOf<String, List<String>>()

    init {
        // ---- Load encoder.json / vocab.json --------------------------------
        val vocabJson = assets.open("encoder.json").bufferedReader().readText()
        val jsonObj   = JSONObject(vocabJson)
        for (key in jsonObj.keys()) {
            val id = jsonObj.getInt(key)
            encoder[key] = id
            decoder[id]  = key
        }

        // ---- Load vocab.bpe (merge rules) ----------------------------------
        assets.open("vocab.bpe").bufferedReader().use { reader ->
            var rank = 0
            reader.lineSequence().drop(1).forEach { line ->   // skip header comment
                val parts = line.trim().split(" ")
                if (parts.size == 2) {
                    bpeRanks[Pair(parts[0], parts[1])] = rank++
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /** Encode a string into a list of GPT-2 token IDs. */
    fun encode(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        for (word in gpt2Pretokenise(text)) {
            // Convert each byte of the UTF-8 word to its unicode proxy char
            val wordBytes   = word.toByteArray(Charsets.UTF_8)
            val wordEncoded = wordBytes.joinToString("") { b ->
                byteEncoder[b.toInt() and 0xFF]!!
            }
            // Apply BPE and convert each merged token to an ID
            for (token in bpe(wordEncoded)) {
                tokens.add(encoder[token] ?: encoder["<|unk|>"] ?: 0)
            }
        }
        return tokens
    }

    /** Decode a list of GPT-2 token IDs back into a UTF-8 string. */
    fun decode(ids: List<Int>): String {
        val text  = ids.joinToString("") { id -> decoder[id] ?: "" }
        val bytes = text.map { ch -> byteDecoder[ch.toString()]?.toByte() ?: '?'.code.toByte() }.toByteArray()
        return String(bytes, Charsets.UTF_8)
    }

    // ------------------------------------------------------------------
    // GPT-2 regex pretokeniser
    // ------------------------------------------------------------------

    /**
     * Split `text` using the GPT-2 pretokenisation pattern:
     *   /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/
     *
     * Java regex doesn't support \p{L} the same way as Python's `regex`
     * library, but the Unicode categories work in java.util.regex since Java 7.
     */
    private val GPT2_PATTERN: Regex = Regex(
        """'(?:[sdmt]|ll|ve|re)|""" +
        """ ?\p{L}+|""" +
        """ ?\p{N}+|""" +
        """ ?[^\s\p{L}\p{N}]+|""" +
        """\s+(?!\S)|""" +
        """\s+"""
    )

    private fun gpt2Pretokenise(text: String): List<String> =
        GPT2_PATTERN.findAll(text).map { it.value }.toList()

    // ------------------------------------------------------------------
    // BPE merge
    // ------------------------------------------------------------------

    private fun bpe(word: String): List<String> {
        bpeCache[word]?.let { return it }

        var symbols = word.map { it.toString() }.toMutableList()
        if (symbols.size <= 1) {
            bpeCache[word] = symbols
            return symbols
        }

        while (true) {
            // Find the pair with the lowest rank
            var bestPair: Pair<String, String>? = null
            var bestRank  = Int.MAX_VALUE

            for (i in 0 until symbols.size - 1) {
                val pair = Pair(symbols[i], symbols[i + 1])
                val rank = bpeRanks[pair]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestPair = pair
                }
            }

            if (bestPair == null) break  // no more merges applicable

            // Merge all occurrences of bestPair
            val (first, second) = bestPair
            val merged = mutableListOf<String>()
            var i = 0
            while (i < symbols.size) {
                if (i < symbols.size - 1 && symbols[i] == first && symbols[i + 1] == second) {
                    merged.add(first + second)
                    i += 2
                } else {
                    merged.add(symbols[i])
                    i++
                }
            }
            symbols = merged
        }

        bpeCache[word] = symbols
        return symbols
    }

    // ------------------------------------------------------------------
    // bytes_to_unicode mapping (identical to the Python original)
    // ------------------------------------------------------------------

    /**
     * Returns a map from byte value (Int 0–255) to a single-character String.
     * Printable ASCII chars map to themselves; everything else maps to a
     * unicode character in a higher plane so the vocab stays printable.
     */
    private fun buildByteEncoder(): Map<Int, String> {
        val bs = mutableListOf<Int>()
        // '!' to '~' (33–126): standard printable ASCII
        bs.addAll(33..126)
        // '¡' to '¬' (161–172)
        bs.addAll(161..172)
        // '®' to 'ÿ' (174–255)
        bs.addAll(174..255)

        val cs = bs.map { it }.toMutableList()
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        return bs.zip(cs).associate { (b, c) -> b to c.toChar().toString() }
    }
}
