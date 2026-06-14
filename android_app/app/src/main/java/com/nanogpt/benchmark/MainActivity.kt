package com.nanogpt.benchmark

import android.app.Activity
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : Activity() {
    private lateinit var promptInput: EditText
    private lateinit var maxTokensInput: EditText
    private lateinit var runButton: Button
    private lateinit var resultPanel: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
    }

    private fun buildUi() {
        val density = resources.displayMetrics.density
        fun dp(value: Int): Int = (value * density).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        root.addView(TextView(this).apply {
            text = "nanoGPT Android Benchmark"
            textSize = 24f
            gravity = Gravity.CENTER_HORIZONTAL
        })

        promptInput = EditText(this).apply {
            hint = "Prompt"
            minLines = 3
            setText("To be, or not to be")
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        }
        root.addView(promptInput)

        maxTokensInput = EditText(this).apply {
            hint = "max_new_tokens"
            setText("32")
            inputType = InputType.TYPE_CLASS_NUMBER
        }
        root.addView(maxTokensInput)

        runButton = Button(this).apply {
            text = "Run benchmark"
            setOnClickListener { runBenchmark() }
        }
        root.addView(runButton)

        resultPanel = TextView(this).apply {
            textSize = 16f
            setPadding(0, dp(16), 0, 0)
            text = startupMessage()
        }
        root.addView(resultPanel)

        setContentView(ScrollView(this).apply { addView(root) })
    }

    private fun runBenchmark() {
        val prompt = promptInput.text.toString()
        val maxNewTokens = maxTokensInput.text.toString().toIntOrNull()?.coerceAtLeast(1) ?: 32
        runButton.isEnabled = false
        resultPanel.text = "Running warmup and measured generation..."

        thread(name = "nanogpt-benchmark") {
            val output = try {
                createModelRunner().use { runner ->
                    val result = BenchmarkRunner(runner).run(prompt, maxNewTokens)
                    formatResult(result)
                }
            } catch (t: Throwable) {
                "Benchmark failed:\n${t.message}\n\n${startupMessage()}"
            }

            runOnUiThread {
                resultPanel.text = output
                runButton.isEnabled = true
            }
        }
    }

    private fun createModelRunner(): ModelRunner {
        val assets = assets.list("")?.toSet().orEmpty()
        return when {
            "model.onnx" in assets -> OrtModelRunner(this)
            "model.pte" in assets -> ExecuTorchModelRunner(this)
            else -> error("Add model.onnx or model.pte to app/src/main/assets before running.")
        }
    }

    private fun formatResult(result: GenerationResult): String = buildString {
        appendLine("Generated text:")
        appendLine(result.generatedText)
        appendLine()
        appendLine("Prompt token count: ${result.promptTokenCount}")
        appendLine("Generated token count: ${result.generatedTokenCount}")
        appendMetric("TTFT ms", result.ttftMs)
        appendMetric("TPOT ms/token", result.tpotMs)
        appendMetric("Total generation time ms", result.totalGenerationMs)
        appendMetric("tokens/sec", result.tokensPerSecond)
        appendMetric("Java/Kotlin heap MB", result.javaHeapMb)
        appendMetric("native heap MB", result.nativeHeapMb)
        appendMetric("Android PSS MB", result.androidPssMb)
    }

    private fun StringBuilder.appendMetric(label: String, value: Double) {
        appendLine("$label: ${String.format(Locale.US, "%.2f", value)}")
    }

    private fun startupMessage(): String = """
        Copy an exported model into app/src/main/assets before running:
        - model.onnx for ONNX Runtime Mobile
        - model.pte for the ExecuTorch placeholder path

        Also replace meta.json with the tokenizer metadata that matches the exported nanoGPT checkpoint.
    """.trimIndent()
}
