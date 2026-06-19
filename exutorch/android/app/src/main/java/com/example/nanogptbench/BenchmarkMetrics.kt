package com.example.nanogptbench

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import kotlin.math.roundToLong

/**
 * Collects on-device LLM inference metrics:
 *
 *  - TTFT (Time-to-First-Token): wall-clock time from the START of the
 *    prefill forward-pass until the first generated token is available.
 *
 *  - Decode latency: per-token wall-clock times for all subsequent steps.
 *    Reports min / median / max / average and a compact histogram.
 *
 *  - Memory: process-level PSS (Proportional Set Size) snapshot before and
 *    after inference.  Delta = peak increase attributed to inference.
 *
 *  - Energy: estimated µWh consumed during the entire generation.
 *    Uses BATTERY_PROPERTY_ENERGY_COUNTER when available (Qualcomm/MTK SoCs
 *    expose this); falls back to charge-counter × nominal voltage.
 *
 * Usage
 * -----
 *   val metrics = BenchmarkMetrics(context)
 *   metrics.startSession()
 *   // ... run prefill ...
 *   metrics.recordTTFT()               // call immediately after first token
 *   // ... decode loop ...
 *   metrics.startDecodeStep()
 *   // ... run one decode step ...
 *   metrics.endDecodeStep()            // call after each step
 *   val report = metrics.endSession()  // final snapshot + report
 */
class BenchmarkMetrics(private val context: Context) {

    // ------------------------------------------------------------------
    // Internal state
    // ------------------------------------------------------------------

    private var sessionStartNs:    Long = 0L
    private var prefillStartNs:    Long = 0L
    private var ttftNs:            Long = 0L          // 0 = not yet recorded

    private var decodeStepStartNs: Long = 0L
    private val decodeLatenciesNs  = mutableListOf<Long>()

    private var memBeforePss: Long = 0L               // KB
    private var memPeakPss:   Long = 0L               // KB

    private var energyStartUwh: Long = Long.MIN_VALUE // µWh (MIN_VALUE = unavailable)
    private var energyEndUwh:   Long = Long.MIN_VALUE

    private val batteryManager: BatteryManager =
        context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /** Call once before starting any inference. */
    fun startSession() {
        sessionStartNs    = System.nanoTime()
        prefillStartNs    = sessionStartNs
        ttftNs            = 0L
        decodeLatenciesNs.clear()
        memBeforePss      = currentPssKb()
        memPeakPss        = memBeforePss
        energyStartUwh    = readEnergyUwh()
    }

    /** Call after the first generated token is available (end of prefill). */
    fun recordTTFT() {
        if (ttftNs == 0L) {
            ttftNs = System.nanoTime() - prefillStartNs
        }
        updatePeakMemory()
    }

    /** Call immediately before running a single decode step. */
    fun startDecodeStep() {
        decodeStepStartNs = System.nanoTime()
    }

    /** Call immediately after a single decode step completes. */
    fun endDecodeStep() {
        val latencyNs = System.nanoTime() - decodeStepStartNs
        decodeLatenciesNs.add(latencyNs)
        updatePeakMemory()
    }

    /**
     * Call after all decode steps are finished.
     * Returns a human-readable [Report] with all measured metrics.
     */
    fun endSession(): Report {
        val totalNs    = System.nanoTime() - sessionStartNs
        energyEndUwh   = readEnergyUwh()
        updatePeakMemory()

        val memDeltaKb = maxOf(0L, memPeakPss - memBeforePss)

        val energyUwh: Long? = when {
            energyStartUwh == Long.MIN_VALUE || energyEndUwh == Long.MIN_VALUE -> null
            else -> maxOf(0L, energyStartUwh - energyEndUwh) // energy counter decreases
        }

        return Report(
            ttftMs           = ttftNs / 1_000_000.0,
            decodeLatenciesMs = decodeLatenciesNs.map { it / 1_000_000.0 },
            memBeforeKb      = memBeforePss,
            memPeakDeltaKb   = memDeltaKb,
            energyUwh        = energyUwh,
            totalMs          = totalNs / 1_000_000.0,
            tokenCount        = decodeLatenciesNs.size,
        )
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    private fun currentPssKb(): Long {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss.toLong()
    }

    private fun updatePeakMemory() {
        val pss = currentPssKb()
        if (pss > memPeakPss) memPeakPss = pss
    }

    /**
     * Read energy from BatteryManager.
     *
     * BATTERY_PROPERTY_ENERGY_COUNTER returns µWh if the hardware supports it
     * (returns Long.MIN_VALUE if not).  As a fallback we derive µWh from the
     * charge counter (µAh) × nominal voltage read from the battery sticky
     * intent.
     */
    private fun readEnergyUwh(): Long {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            val energy = batteryManager.getLongProperty(
                BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER
            )
            if (energy != Long.MIN_VALUE) return energy  // µWh directly

            // Fallback: charge counter (µAh) × voltage (mV) / 1000 → µWh
            val chargeUah = batteryManager.getLongProperty(
                BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER
            )
            if (chargeUah != Long.MIN_VALUE) {
                val voltMv = readBatteryVoltageMv()
                if (voltMv > 0) {
                    return chargeUah * voltMv / 1000L   // µAh × mV / 1000 = µWh
                }
            }
        }
        return Long.MIN_VALUE
    }

    private fun readBatteryVoltageMv(): Int {
        val intent = context.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )
        return intent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1
    }

    // ------------------------------------------------------------------
    // Report data class
    // ------------------------------------------------------------------

    data class Report(
        val ttftMs:            Double,
        val decodeLatenciesMs: List<Double>,
        val memBeforeKb:       Long,
        val memPeakDeltaKb:    Long,
        val energyUwh:         Long?,
        val totalMs:           Double,
        val tokenCount:        Int,
    ) {
        val avgDecodeMs: Double
            get() = if (decodeLatenciesMs.isEmpty()) 0.0
                    else decodeLatenciesMs.average()

        val minDecodeMs: Double
            get() = decodeLatenciesMs.minOrNull() ?: 0.0

        val maxDecodeMs: Double
            get() = decodeLatenciesMs.maxOrNull() ?: 0.0

        val medianDecodeMs: Double
            get() {
                if (decodeLatenciesMs.isEmpty()) return 0.0
                val sorted = decodeLatenciesMs.sorted()
                val mid    = sorted.size / 2
                return if (sorted.size % 2 == 0)
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                else
                    sorted[mid]
            }

        val tokensPerSecond: Double
            get() = if (avgDecodeMs <= 0.0) 0.0 else 1000.0 / avgDecodeMs

        val memPeakDeltaMb: Double
            get() = memPeakDeltaKb / 1024.0

        /** Compact 8-bucket histogram of decode latencies. */
        val latencyHistogram: String
            get() {
                if (decodeLatenciesMs.isEmpty()) return "(no decode steps)"
                val min    = decodeLatenciesMs.min()
                val max    = decodeLatenciesMs.max()
                val nBins  = 8
                val width  = (max - min).coerceAtLeast(1.0) / nBins
                val counts = IntArray(nBins)
                for (v in decodeLatenciesMs) {
                    val idx = ((v - min) / width).toInt().coerceIn(0, nBins - 1)
                    counts[idx]++
                }
                val maxCount = counts.max().coerceAtLeast(1)
                return buildString {
                    for (i in 0 until nBins) {
                        val lo = min + i * width
                        val hi = lo + width
                        val bar = "█".repeat((counts[i] * 20 / maxCount).coerceAtLeast(
                            if (counts[i] > 0) 1 else 0
                        ))
                        appendLine("%5.1f–%5.1f ms | %-20s %d".format(lo, hi, bar, counts[i]))
                    }
                }.trimEnd()
            }

        override fun toString(): String = buildString {
            appendLine("=== NanoGPT Benchmark Report ===")
            appendLine("TTFT                : %.1f ms".format(ttftMs))
            appendLine("Decode tokens       : $tokenCount")
            appendLine("Avg decode latency  : %.1f ms/tok".format(avgDecodeMs))
            appendLine("Min / Median / Max  : %.1f / %.1f / %.1f ms".format(
                minDecodeMs, medianDecodeMs, maxDecodeMs))
            appendLine("Throughput          : %.2f tokens/s".format(tokensPerSecond))
            appendLine("Total time          : %.1f ms".format(totalMs))
            appendLine("Memory (PSS) delta  : %.1f MB".format(memPeakDeltaMb))
            if (energyUwh != null)
                appendLine("Energy consumed     : %.1f µWh  (%.3f mWh)".format(
                    energyUwh.toDouble(), energyUwh / 1000.0))
            else
                appendLine("Energy consumed     : N/A (hardware counter unavailable)")
            appendLine()
            appendLine("Decode latency histogram:")
            appendLine(latencyHistogram)
        }
    }
}
