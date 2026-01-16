package com.nanogpt.executorch.template

class TokenSampler {
    fun argmax(logits: FloatArray): Int {
        var bestIndex = 0
        var bestValue = Float.NEGATIVE_INFINITY
        for (i in logits.indices) {
            val value = logits[i]
            if (value > bestValue) {
                bestValue = value
                bestIndex = i
            }
        }
        return bestIndex
    }
}
