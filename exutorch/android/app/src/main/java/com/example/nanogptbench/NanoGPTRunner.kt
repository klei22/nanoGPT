package com.example.nanogptbench

import android.content.res.AssetManager
import com.facebook.executorch.EValue
import com.facebook.executorch.Module
import com.facebook.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp
import kotlin.math.ln

/**
 * Wraps the ExecuTorch [Module] to run nanoGPT token-by-token generation.
 *
 * Architecture
 * ------------
 * The model is the "sliding window" variant exported by export_nanogpt_android.py:
 *   - Input : LongTensor (1, seq_len)  – full context up to block_size tokens
 *   - Output: FloatTensor (1, 1, vocab_size) – logits for the next token
 *
 * This maps naturally to the benchmark loop:
 *   1. prefill   – feed all prompt tokens, measure TTFT
 *   2. decode    – append the sampled token and re-run, measure each step
 *
 * Token sampling
 * --------------
 * Three strategies are provided: greedy, top-k, and top-p (nucleus).
 * Defaults to top-k = 40, temperature = 0.8.
 */
class NanoGPTRunner(assets: AssetManager) {

    companion object {
        private const val ENDOFTEXT_TOKEN = 50256
        private const val MAX_CONTEXT     = 1024
    }

    // ExecuTorch module (loaded lazily from assets)
    private var module: Module? = null

    // Model input/output shape info (populated after load)
    var vocabSize:  Int = 50257
        private set
    var blockSize:  Int = 1024
        private set

    // ---------------------------------------------------------------
    // Loading
    // ---------------------------------------------------------------

    /**
     * Copy the .pte asset to a temp file (ExecuTorch needs a filesystem path)
     * and load it.
     */
    fun loadModel(assets: AssetManager, assetName: String) {
        // ExecuTorch's Module.load() requires a real path, not an asset URI
        val tmpFile = File.createTempFile("nanogpt_", ".pte")
        tmpFile.deleteOnExit()
        assets.open(assetName).use { input ->
            FileOutputStream(tmpFile).use { output -> input.copyTo(output) }
        }
        module = Module.load(tmpFile.absolutePath)
    }

    fun isLoaded(): Boolean = module != null

    // ---------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------

    /**
     * Run a full forward pass over [contextTokens].
     * Returns the logits FloatArray of size [vocabSize].
     *
     * This is the function wrapped by the benchmark for both prefill
     * (first call) and decode (subsequent calls).
     */
    fun forward(contextTokens: LongArray): FloatArray {
        val mod = requireNotNull(module) { "Model not loaded – call loadModel() first" }

        // Truncate to block_size if needed
        val tokens = if (contextTokens.size > MAX_CONTEXT)
            contextTokens.copyOfRange(contextTokens.size - MAX_CONTEXT, contextTokens.size)
        else
            contextTokens

        val inputTensor = Tensor.fromBlob(tokens, longArrayOf(1L, tokens.size.toLong()))
        val outputs     = mod.forward(EValue.from(inputTensor))
        val logitsTensor: Tensor = outputs[0].toTensor()
        return logitsTensor.dataAsFloatArray
    }

    // ---------------------------------------------------------------
    // Sampling
    // ---------------------------------------------------------------

    /** Greedy argmax. */
    fun sampleGreedy(logits: FloatArray): Int =
        logits.indices.maxByOrNull { logits[it] } ?: 0

    /**
     * Top-k sampling with temperature.
     *
     * @param logits   Raw logits from the model
     * @param topK     Number of top candidates to sample from
     * @param temperature  Softmax temperature (< 1.0 = sharper, > 1.0 = flatter)
     */
    fun sampleTopK(logits: FloatArray, topK: Int = 40, temperature: Float = 0.8f): Int {
        // Scale by temperature
        val scaled = FloatArray(logits.size) { logits[it] / temperature }

        // Get top-k indices (sorted descending)
        val topKIndices = (scaled.indices)
            .sortedByDescending { scaled[it] }
            .take(topK.coerceAtMost(scaled.size))

        // Softmax over top-k
        val topKLogits = FloatArray(topKIndices.size) { scaled[topKIndices[it]] }
        val maxLogit   = topKLogits.max()
        val exps       = FloatArray(topKLogits.size) { exp((topKLogits[it] - maxLogit).toDouble()).toFloat() }
        val sumExps    = exps.sum()
        val probs      = FloatArray(exps.size) { exps[it] / sumExps }

        // Multinomial sample
        val r = Math.random().toFloat()
        var cumulative = 0f
        for (i in probs.indices) {
            cumulative += probs[i]
            if (r <= cumulative) return topKIndices[i]
        }
        return topKIndices.last()
    }

    /**
     * Top-p (nucleus) sampling with temperature.
     *
     * @param p  Probability mass to sample from (0.0–1.0, typical: 0.9)
     */
    fun sampleTopP(logits: FloatArray, p: Float = 0.9f, temperature: Float = 0.8f): Int {
        val scaled      = FloatArray(logits.size) { logits[it] / temperature }
        val sortedIdxs  = scaled.indices.sortedByDescending { scaled[it] }
        val maxLogit    = scaled[sortedIdxs[0]]
        val exps        = FloatArray(scaled.size) { exp((scaled[it] - maxLogit).toDouble()).toFloat() }
        val sumExps     = exps.sum()

        var cumProb     = 0f
        val nucleusIdxs = mutableListOf<Int>()
        for (idx in sortedIdxs) {
            cumProb += exps[idx] / sumExps
            nucleusIdxs.add(idx)
            if (cumProb >= p) break
        }

        // Renormalise and sample
        val nucleusExps = FloatArray(nucleusIdxs.size) { exps[nucleusIdxs[it]] }
        val nucleusSum  = nucleusExps.sum()
        val probs       = FloatArray(nucleusExps.size) { nucleusExps[it] / nucleusSum }

        val r = Math.random().toFloat()
        var cumulative = 0f
        for (i in probs.indices) {
            cumulative += probs[i]
            if (r <= cumulative) return nucleusIdxs[i]
        }
        return nucleusIdxs.last()
    }

    // ---------------------------------------------------------------
    // Full generation loop with benchmark metrics
    // ---------------------------------------------------------------

    /**
     * Generate [maxNewTokens] tokens starting from [promptTokens].
     *
     * Calls back into [onToken] with each decoded string piece and the raw
     * token ID as generation proceeds.
     *
     * @param tokenizer    BPETokenizer for converting IDs → text
     * @param metrics      BenchmarkMetrics collector (session must be started)
     * @param promptTokens Integer token IDs for the prompt
     * @param maxNewTokens Maximum tokens to generate
     * @param onToken      Callback (tokenString, tokenId) called per generated token
     * @param onDone       Called when generation finishes
     */
    fun generate(
        tokenizer:     BPETokenizer,
        metrics:       BenchmarkMetrics,
        promptTokens:  LongArray,
        maxNewTokens:  Int = 200,
        onToken:       (String, Int) -> Unit = { _, _ -> },
        onDone:        (BenchmarkMetrics.Report) -> Unit = {},
    ) {
        val context = promptTokens.toMutableList()

        // ---- Prefill (TTFT) -------------------------------------------
        var logits = forward(context.toLongArray())
        metrics.recordTTFT()

        var nextToken = sampleTopK(logits)
        if (nextToken == ENDOFTEXT_TOKEN) {
            onDone(metrics.endSession())
            return
        }
        onToken(tokenizer.decode(listOf(nextToken)), nextToken)
        context.add(nextToken.toLong())

        // ---- Decode loop -----------------------------------------------
        for (step in 1 until maxNewTokens) {
            metrics.startDecodeStep()
            logits    = forward(context.toLongArray())
            metrics.endDecodeStep()

            nextToken = sampleTopK(logits)
            if (nextToken == ENDOFTEXT_TOKEN) break

            onToken(tokenizer.decode(listOf(nextToken)), nextToken)
            context.add(nextToken.toLong())

            // Slide window
            if (context.size > MAX_CONTEXT) {
                context.removeAt(0)
            }
        }

        onDone(metrics.endSession())
    }

    fun close() {
        module?.let {
            // ExecuTorch Module does not implement Closeable in 0.4, but
            // dereferencing it allows GC to release native memory
        }
        module = null
    }
}
