package com.example.nanogpt

import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    private lateinit var module: Module

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val statusView = TextView(this)
        statusView.text = "Running ExecuTorch nanoGPT..."
        setContentView(statusView)

        val manifest = readManifestJson()
        val blockSize = manifest.optJSONObject("model_config")?.optInt("block_size", 128) ?: 128
        val vocabSizeManifest = manifest.optJSONObject("model_config")?.optInt("vocab_size", -1) ?: -1

        val modelPath = assetFilePath("nanogpt_xnnpack.pte")
        module = Module.load(modelPath)

        val promptTokens = readPromptTokens()
        val tokens = promptTokens.toMutableList()
        val maxNewTokens = 16

        var timeToFirstTokenMs = 0L
        val decodeTimes = mutableListOf<Long>()

        for (step in 0 until maxNewTokens) {
            val inputTokens = tokens.takeLast(blockSize).toLongArray()
            val inputTensor = Tensor.fromBlob(inputTokens, longArrayOf(1, inputTokens.size.toLong()))
            val startNs = SystemClock.elapsedRealtimeNanos()

            val output = module.forward(EValue.from(inputTensor)).toTensor()
            val logits = output.dataAsFloatArray
            val vocabSize = if (vocabSizeManifest > 0) {
                vocabSizeManifest
            } else {
                logits.size / inputTokens.size
            }

            val lastOffset = (inputTokens.size - 1) * vocabSize
            var maxIdx = 0
            var maxVal = Float.NEGATIVE_INFINITY
            for (i in 0 until vocabSize) {
                val value = logits[lastOffset + i]
                if (value > maxVal) {
                    maxVal = value
                    maxIdx = i
                }
            }

            val endNs = SystemClock.elapsedRealtimeNanos()
            val elapsedMs = (endNs - startNs) / 1_000_000
            if (step == 0) {
                timeToFirstTokenMs = elapsedMs
            } else {
                decodeTimes.add(elapsedMs)
            }

            tokens.add(maxIdx.toLong())
        }

        val avgDecodeMs = if (decodeTimes.isNotEmpty()) {
            decodeTimes.average()
        } else {
            0.0
        }

        val report = buildString {
            appendLine("TTFT ms: $timeToFirstTokenMs")
            appendLine("Avg decode ms/token: ${"%.2f".format(avgDecodeMs)}")
            appendLine("Generated tokens: ${tokens.joinToString(", ")}")
        }

        Log.i("NanoGPT", report)
        statusView.text = report
    }

    private fun readManifestJson(): JSONObject {
        val text = assets.open("manifest.json").bufferedReader().use { it.readText() }
        return JSONObject(text)
    }

    private fun readPromptTokens(): List<Long> {
        val text = assets.open("prompt_tokens.txt").bufferedReader().use { it.readText() }
        return text.split(",")
            .mapNotNull { it.trim().takeIf { token -> token.isNotEmpty() } }
            .map { it.toLong() }
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        return file.absolutePath
    }
}
