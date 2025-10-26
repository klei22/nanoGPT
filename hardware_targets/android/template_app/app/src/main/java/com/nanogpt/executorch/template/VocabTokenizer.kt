package com.nanogpt.executorch.template

import android.content.res.AssetManager
import java.io.BufferedReader
import java.io.InputStreamReader
import org.json.JSONObject

class VocabTokenizer(
    private val tokenToId: Map<String, Int>,
    private val idToToken: Map<Int, String>,
    val endOfTextToken: Int
) {
    fun encode(text: String): List<Int> {
        val result = ArrayList<Int>(text.length)
        for (char in text) {
            val key = char.toString()
            val tokenId = tokenToId[key]
            if (tokenId != null) {
                result.add(tokenId)
            } else {
                tokenToId[UNKNOWN_TOKEN]?.let { result.add(it) }
            }
        }
        return result
    }

    fun decode(tokens: List<Int>): String {
        val builder = StringBuilder(tokens.size)
        for (token in tokens) {
            val piece = idToToken[token]
            if (piece != null) {
                builder.append(piece)
            }
        }
        return builder.toString()
    }

    companion object {
        private const val UNKNOWN_TOKEN = "<unk>"

        fun fromAssets(assetManager: AssetManager, fileName: String): VocabTokenizer {
            assetManager.open(fileName).use { inputStream ->
                val reader = BufferedReader(InputStreamReader(inputStream))
                val content = buildString {
                    var line = reader.readLine()
                    while (line != null) {
                        append(line)
                        line = reader.readLine()
                    }
                }
                val json = JSONObject(content)
                val mapping = mutableMapOf<String, Int>()
                val reverse = mutableMapOf<Int, String>()
                val keys = json.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    val value = json.getInt(key)
                    mapping[key] = value
                    reverse[value] = key
                }
                val endToken = mapping["<|endoftext|>"] ?: -1
                return VocabTokenizer(mapping, reverse, endToken)
            }
        }
    }
}
