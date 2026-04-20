package com.example.nanogptbench

import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.nanogptbench.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

/**
 * Main benchmark activity.
 *
 * Layout (activity_main.xml)
 * --------------------------
 *   - Spinner: model selector (fp32 / int8)
 *   - EditText: prompt
 *   - SeekBar + label: max tokens (1–200)
 *   - Button: "Run Benchmark"
 *   - ProgressBar (indeterminate)
 *   - Scrollable generated text box
 *   - CardViews for each metric:
 *       TTFT | Avg decode | Tokens/s | Memory delta | Energy
 *   - TextView: full histogram + report
 *
 * All inference runs on [Dispatchers.IO]; UI updates post back to Main.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding:    ActivityMainBinding
    private lateinit var tokenizer:  BPETokenizer
    private lateinit var runner:     NanoGPTRunner
    private lateinit var metrics:    BenchmarkMetrics

    // Available model assets (must be present in app/src/main/assets/)
    private val MODEL_OPTIONS = listOf(
        "nanogpt_fp32.pte"  to "GPT-2 FP32 (XNNPack)",
        "nanogpt_int8.pte"  to "GPT-2 INT8 (XNNPack)",
    )

    private var selectedModelAsset = MODEL_OPTIONS[0].first
    private var maxTokens           = 100

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        tokenizer = BPETokenizer(assets)
        runner    = NanoGPTRunner(assets)
        metrics   = BenchmarkMetrics(this)

        setupModelSpinner()
        setupTokenSlider()
        setupRunButton()
    }

    override fun onDestroy() {
        super.onDestroy()
        runner.close()
    }

    // ------------------------------------------------------------------
    // UI setup
    // ------------------------------------------------------------------

    private fun setupModelSpinner() {
        val labels  = MODEL_OPTIONS.map { it.second }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, labels)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerModel.adapter = adapter
        binding.spinnerModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                selectedModelAsset = MODEL_OPTIONS[pos].first
                // Force re-load next run
                runner.close()
            }
            override fun onNothingSelected(p: AdapterView<*>?) = Unit
        }
    }

    private fun setupTokenSlider() {
        binding.seekbarTokens.max      = 199
        binding.seekbarTokens.progress = 99   // default 100 tokens
        binding.labelTokens.text       = "Max tokens: 100"

        binding.seekbarTokens.setOnSeekBarChangeListener(object :
            android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: android.widget.SeekBar, progress: Int, fromUser: Boolean) {
                maxTokens = progress + 1
                binding.labelTokens.text = "Max tokens: $maxTokens"
            }
            override fun onStartTrackingTouch(sb: android.widget.SeekBar) = Unit
            override fun onStopTrackingTouch(sb: android.widget.SeekBar)  = Unit
        })
    }

    private fun setupRunButton() {
        binding.btnRun.setOnClickListener {
            val prompt = binding.editPrompt.text.toString().trim()
            if (prompt.isEmpty()) {
                binding.editPrompt.error = "Enter a prompt"
                return@setOnClickListener
            }
            runBenchmark(prompt)
        }
    }

    // ------------------------------------------------------------------
    // Benchmark execution
    // ------------------------------------------------------------------

    private fun runBenchmark(prompt: String) {
        setUiBusy(true)
        resetMetricCards()

        lifecycleScope.launch(Dispatchers.IO) {
            // ---- Load model if needed -----------------------------------
            if (!runner.isLoaded()) {
                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = "Loading model…"
                }
                try {
                    runner.loadModel(assets, selectedModelAsset)
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        binding.tvStatus.text = "Error loading model: ${e.message}"
                        setUiBusy(false)
                    }
                    return@launch
                }
            }

            // ---- Tokenise prompt ----------------------------------------
            val promptTokenIds: LongArray = try {
                tokenizer.encode(prompt).map { it.toLong() }.toLongArray()
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = "Tokenisation error: ${e.message}"
                    setUiBusy(false)
                }
                return@launch
            }

            withContext(Dispatchers.Main) {
                binding.tvStatus.text = "Running… (${promptTokenIds.size} prompt tokens)"
                binding.tvGenerated.text = prompt
            }

            // ---- Run inference with metrics ------------------------------
            val generatedText = StringBuilder()
            metrics.startSession()

            try {
                runner.generate(
                    tokenizer    = tokenizer,
                    metrics      = metrics,
                    promptTokens = promptTokenIds,
                    maxNewTokens = maxTokens,
                    onToken      = { piece, _ ->
                        generatedText.append(piece)
                        lifecycleScope.launch(Dispatchers.Main) {
                            binding.tvGenerated.text = prompt + generatedText.toString()
                        }
                    },
                    onDone       = { report ->
                        lifecycleScope.launch(Dispatchers.Main) {
                            displayReport(report)
                            setUiBusy(false)
                        }
                    }
                )
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = "Inference error: ${e.message}"
                    setUiBusy(false)
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // UI update helpers
    // ------------------------------------------------------------------

    private fun setUiBusy(busy: Boolean) {
        binding.btnRun.isEnabled        = !busy
        binding.progressBar.visibility  = if (busy) View.VISIBLE else View.GONE
        binding.spinnerModel.isEnabled  = !busy
        binding.editPrompt.isEnabled    = !busy
        binding.seekbarTokens.isEnabled = !busy
        if (!busy) binding.tvStatus.text = "Ready"
    }

    private fun resetMetricCards() {
        val dash = "—"
        binding.valueTtft.text         = dash
        binding.valueDecodeAvg.text    = dash
        binding.valueThroughput.text   = dash
        binding.valueMemory.text       = dash
        binding.valueEnergy.text       = dash
        binding.tvReport.text          = ""
        binding.tvGenerated.text       = ""
    }

    private fun displayReport(report: BenchmarkMetrics.Report) {
        binding.tvStatus.text = "Done  (${report.tokenCount} tokens generated)"

        binding.valueTtft.text       = "%.1f ms".format(report.ttftMs)
        binding.valueDecodeAvg.text  = "%.1f ms/tok".format(report.avgDecodeMs)
        binding.valueThroughput.text = "%.2f tok/s".format(report.tokensPerSecond)
        binding.valueMemory.text     = "%.1f MB".format(report.memPeakDeltaMb)
        binding.valueEnergy.text     = if (report.energyUwh != null)
            "%.1f µWh".format(report.energyUwh.toDouble())
        else
            "N/A"

        // Full histogram + report in the details section
        binding.tvReport.text = buildString {
            appendLine("Min: %.1f ms   Median: %.1f ms   Max: %.1f ms".format(
                report.minDecodeMs, report.medianDecodeMs, report.maxDecodeMs))
            appendLine()
            appendLine("Decode latency histogram:")
            appendLine(report.latencyHistogram)
            appendLine()
            appendLine("Memory before: %.1f MB  |  Peak delta: +%.1f MB".format(
                report.memBeforeKb / 1024.0, report.memPeakDeltaMb))
            if (report.energyUwh != null)
                appendLine("Energy: %.1f µWh  (%.4f mWh)".format(
                    report.energyUwh.toDouble(), report.energyUwh / 1000.0))
        }
    }
}
