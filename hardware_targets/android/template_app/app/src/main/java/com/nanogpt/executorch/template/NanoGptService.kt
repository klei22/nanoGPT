package com.nanogpt.executorch.template

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class NanoGptService : LifecycleService() {
    private val moduleReady = AtomicBoolean(false)
    private lateinit var module: Module
    private lateinit var tokenizer: VocabTokenizer
    private val sampler = TokenSampler()

    override fun onCreate() {
        super.onCreate()
        startForeground(NOTIFICATION_ID, buildNotification("Initialising"))
        lifecycleScope.launch(Dispatchers.IO) {
            prepareAssets()
            updateNotification("Ready for prompts")
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            return Service.START_NOT_STICKY
        }
        val prompt = intent.getStringExtra(GenerationBroadcastReceiver.EXTRA_PROMPT) ?: "Hello"
        val maxTokens = intent.getIntExtra(GenerationBroadcastReceiver.EXTRA_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        val contextLength = if (intent.hasExtra(GenerationBroadcastReceiver.EXTRA_CONTEXT_LENGTH)) {
            intent.getIntExtra(GenerationBroadcastReceiver.EXTRA_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        } else {
            null
        }
        val sweep = intent.getIntegerArrayListExtra(GenerationBroadcastReceiver.EXTRA_CONTEXT_SWEEP)
        lifecycleScope.launch(Dispatchers.IO) {
            ensureLoaded()
            val contexts = if (sweep.isNullOrEmpty()) {
                listOf(contextLength ?: DEFAULT_CONTEXT_LENGTH)
            } else {
                sweep.toList()
            }
            updateNotification("Running generation for ${contexts.size} context(s)")
            for (ctx in contexts) {
                runGeneration(prompt, maxTokens, ctx)
            }
            updateNotification("Completed latest request")
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        }
        return Service.START_NOT_STICKY
    }

    override fun onBind(intent: Intent): IBinder? {
        super.onBind(intent)
        return null
    }

    private suspend fun ensureLoaded() {
        if (!moduleReady.get()) {
            prepareAssets()
        }
    }

    private suspend fun prepareAssets() {
        if (moduleReady.get()) {
            return
        }
        withContext(Dispatchers.IO) {
            val assetManager = assets
            val pteFile = File(filesDir, PTE_ASSET)
            if (!pteFile.exists()) {
                assetManager.open(PTE_ASSET).use { input ->
                    FileOutputStream(pteFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }
            module = Module.load(pteFile.absolutePath)
            tokenizer = VocabTokenizer.fromAssets(assetManager, VOCAB_ASSET)
            moduleReady.set(true)
            Log.i(TAG, "ExecuTorch module and tokenizer initialised")
        }
    }

    private suspend fun runGeneration(prompt: String, maxTokens: Int, contextLength: Int) {
        val promptTokens = tokenizer.encode(prompt).toMutableList()
        if (promptTokens.isEmpty()) {
            Log.w(TAG, "Prompt produced no tokens; aborting run")
            return
        }
        val contextTokens = promptTokens.takeLast(contextLength).toMutableList()
        val generated = mutableListOf<Int>()

        val inferenceStart = SystemClock.elapsedRealtimeNanos()
        var ttftMillis = 0.0
        var decodeNanos = 0L

        for (i in 0 until maxTokens) {
            val tokenArray = contextTokens.map { it.toLong() }.toLongArray()
            val inputTensor = Tensor.fromBlob(tokenArray, longArrayOf(1, tokenArray.size.toLong()))
            val runStart = SystemClock.elapsedRealtimeNanos()
            val outputs = module.forward(arrayOf(EValue.from(inputTensor)))
            val runEnd = SystemClock.elapsedRealtimeNanos()
            val logitsTensor = outputs[0].toTensor()
            val logits = FloatArray(logitsTensor.numel())
            logitsTensor.copyTo(logits)
            val nextToken = sampler.argmax(logits)
            if (i == 0) {
                ttftMillis = (runEnd - inferenceStart) / 1_000_000.0
            }
            decodeNanos += runEnd - runStart
            generated.add(nextToken)
            contextTokens.add(nextToken)
            if (contextTokens.size > contextLength) {
                contextTokens.removeAt(0)
            }
            if (nextToken == tokenizer.endOfTextToken) {
                break
            }
        }

        val decodeMillis = decodeNanos / 1_000_000.0
        val metrics = JSONObject()
        val prefix = "ctx${contextLength}"
        metrics.put(prefix + "_ttft", JSONObject().apply {
            put("tokens", 1)
            put("latency_ms", ttftMillis)
            put("energy_mj", 0.0)
        })
        metrics.put(prefix + "_decode", JSONObject().apply {
            put("tokens", generated.size)
            put("latency_ms", decodeMillis)
            put("energy_mj", 0.0)
        })

        Log.i(TAG, "EXECUTORCH_METRICS_BEGIN${metrics}EXECUTORCH_METRICS_END")
        Log.i(TAG, "Generated (${contextLength} ctx): ${tokenizer.decode(generated)}")
    }

    private fun buildNotification(status: String): Notification {
        val channelId = ensureChannel()
        return NotificationCompat.Builder(this, channelId)
            .setContentTitle("NanoGPT ExecuTorch")
            .setContentText(status)
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(status: String) {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, buildNotification(status))
    }

    private fun ensureChannel(): String {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channelId = "nanogpt_exec"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "NanoGPT ExecuTorch", NotificationManager.IMPORTANCE_LOW)
            manager.createNotificationChannel(channel)
        }
        return channelId
    }

    companion object {
        private const val TAG = "NanoGPTTemplate"
        private const val NOTIFICATION_ID = 42
        private const val PTE_ASSET = "nanogpt.pte"
        private const val VOCAB_ASSET = "vocab.json"
        private const val DEFAULT_MAX_TOKENS = 32
        private const val DEFAULT_CONTEXT_LENGTH = 128
    }
}
