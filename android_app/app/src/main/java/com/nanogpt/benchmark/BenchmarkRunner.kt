package com.nanogpt.benchmark

import android.content.Context
import android.os.Debug
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.nio.LongBuffer
import kotlin.math.max

interface ModelRunner : Closeable {
    fun generate(prompt: String, maxNewTokens: Int): GenerationResult
}

data class GenerationResult(
    val promptTokenCount: Int,
    val generatedTokenCount: Int,
    val generatedText: String,
    val ttftMs: Double,
    val tpotMs: Double,
    val totalGenerationMs: Double,
    val tokensPerSecond: Double,
    val javaHeapMb: Double,
    val nativeHeapMb: Double,
    val androidPssMb: Double
)

class BenchmarkRunner(private val modelRunner: ModelRunner) {
    fun run(prompt: String, maxNewTokens: Int): GenerationResult {
        modelRunner.generate(prompt = "Hello", maxNewTokens = 1)
        return modelRunner.generate(prompt = prompt, maxNewTokens = maxNewTokens)
    }
}

class OrtModelRunner(context: Context) : ModelRunner {
    private val appContext = context.applicationContext
    private val tokenizer = CharTokenizer.load(appContext)
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        val modelFile = copyAssetToCache("model.onnx")
        val options = OrtSession.SessionOptions()
        session = env.createSession(modelFile.absolutePath, options)
        inputName = session.inputNames.firstOrNull()
            ?: error("The ONNX model does not expose any input tensors.")
    }

    override fun generate(prompt: String, maxNewTokens: Int): GenerationResult {
        val safeMaxNewTokens = max(1, maxNewTokens)
        val promptTokens = tokenizer.encode(prompt)
        val generated = mutableListOf<Int>()
        val allTokens = promptTokens.toMutableList()
        val firstInvokeStartNs = System.nanoTime()
        var firstTokenNs = firstInvokeStartNs
        var previousTokenNs = firstInvokeStartNs
        var postFirstTokenNs = 0L

        repeat(safeMaxNewTokens) { index ->
            val logits = runForFinalLogits(allTokens)
            val nextToken = argmax(logits)
            allTokens += nextToken
            generated += nextToken
            val now = System.nanoTime()
            if (index == 0) {
                firstTokenNs = now
            } else {
                postFirstTokenNs += now - previousTokenNs
            }
            previousTokenNs = now
        }

        val totalNs = previousTokenNs - firstInvokeStartNs
        val ttftMs = nsToMs(firstTokenNs - firstInvokeStartNs)
        val tpotMs = if (generated.size > 1) nsToMs(postFirstTokenNs) / (generated.size - 1) else 0.0
        val totalMs = nsToMs(totalNs)
        val memory = MemoryStats.capture()
        return GenerationResult(
            promptTokenCount = promptTokens.size,
            generatedTokenCount = generated.size,
            generatedText = tokenizer.decode(generated),
            ttftMs = ttftMs,
            tpotMs = tpotMs,
            totalGenerationMs = totalMs,
            tokensPerSecond = if (totalMs > 0.0) generated.size / (totalMs / 1000.0) else 0.0,
            javaHeapMb = memory.javaHeapMb,
            nativeHeapMb = memory.nativeHeapMb,
            androidPssMb = memory.androidPssMb
        )
    }

    private fun runForFinalLogits(tokens: List<Int>): FloatArray {
        val shape = longArrayOf(1L, tokens.size.toLong())
        val ids = LongArray(tokens.size) { tokens[it].toLong() }
        val tensor = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape)
        try {
            val result = session.run(mapOf(inputName to tensor))
            try {
                val output = result.get(0).value
                return extractFinalLogits(output)
            } finally {
                result.close()
            }
        } finally {
            tensor.close()
        }
    }

    private fun extractFinalLogits(output: Any): FloatArray {
        @Suppress("UNCHECKED_CAST")
        return when (output) {
            is Array<*> -> {
                val first = output.firstOrNull() ?: error("Model returned an empty logits tensor.")
                when (first) {
                    is Array<*> -> first.lastOrNull() as? FloatArray
                        ?: error("Expected ONNX logits with shape [batch, tokens, vocab].")
                    is FloatArray -> first
                    else -> error("Unsupported ONNX output element: ${first::class.java.name}")
                }
            }
            is FloatArray -> output
            else -> error("Unsupported ONNX output: ${output::class.java.name}")
        }
    }

    private fun copyAssetToCache(assetName: String): File {
        val outFile = File(appContext.filesDir, assetName)
        appContext.assets.open(assetName).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outFile
    }

    override fun close() {
        session.close()
    }
}

class ExecuTorchModelRunner(context: Context) : ModelRunner {
    init {
        context.applicationContext.assets.open("model.pte").close()
    }

    override fun generate(prompt: String, maxNewTokens: Int): GenerationResult {
        error("ExecuTorch .pte execution is a placeholder. Add the ExecuTorch Android dependency and tensor binding code for your exported model signature, or use model.onnx with OrtModelRunner.")
    }

    override fun close() = Unit
}

private data class MemoryStats(
    val javaHeapMb: Double,
    val nativeHeapMb: Double,
    val androidPssMb: Double
) {
    companion object {
        fun capture(): MemoryStats {
            val memoryInfo = Debug.MemoryInfo()
            Debug.getMemoryInfo(memoryInfo)
            val javaHeapBytes = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val nativeHeapBytes = Debug.getNativeHeapAllocatedSize()
            return MemoryStats(
                javaHeapMb = bytesToMb(javaHeapBytes),
                nativeHeapMb = bytesToMb(nativeHeapBytes),
                androidPssMb = memoryInfo.totalPss / 1024.0
            )
        }
    }
}

class CharTokenizer private constructor(
    private val stoi: Map<String, Int>,
    private val itos: Map<Int, String>
) {
    fun encode(text: String): List<Int> = text.map { ch ->
        stoi[ch.toString()] ?: stoi[" "] ?: 0
    }

    fun decode(tokens: List<Int>): String = buildString {
        tokens.forEach { append(itos[it] ?: "") }
    }

    companion object {
        fun load(context: Context): CharTokenizer {
            val assetName = when {
                assetExists(context, "tokenizer.json") -> "tokenizer.json"
                assetExists(context, "meta.json") -> "meta.json"
                else -> error("Add tokenizer.json or meta.json to app/src/main/assets.")
            }
            val json = context.assets.open(assetName).bufferedReader().use { it.readText() }
            val root = JSONObject(json)
            val stoiJson = root.optJSONObject("stoi")
                ?: root.optJSONObject("model")?.optJSONObject("vocab")
                ?: error("$assetName must contain a top-level stoi object or model.vocab object.")
            val stoi = mutableMapOf<String, Int>()
            val keys = stoiJson.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                stoi[key] = stoiJson.getInt(key)
            }
            val explicitItos = root.optJSONObject("itos")
            val itos = if (explicitItos != null) {
                val decoded = mutableMapOf<Int, String>()
                val itosKeys = explicitItos.keys()
                while (itosKeys.hasNext()) {
                    val key = itosKeys.next()
                    decoded[key.toInt()] = explicitItos.getString(key)
                }
                decoded
            } else {
                stoi.entries.associate { (token, id) -> id to token }
            }
            return CharTokenizer(stoi, itos)
        }

        private fun assetExists(context: Context, assetName: String): Boolean =
            context.assets.list("")?.contains(assetName) == true
    }
}

private fun argmax(values: FloatArray): Int {
    var bestIndex = 0
    var bestValue = Float.NEGATIVE_INFINITY
    values.forEachIndexed { index, value ->
        if (value > bestValue) {
            bestValue = value
            bestIndex = index
        }
    }
    return bestIndex
}

private fun nsToMs(ns: Long): Double = ns / 1_000_000.0
private fun bytesToMb(bytes: Long): Double = bytes / (1024.0 * 1024.0)
