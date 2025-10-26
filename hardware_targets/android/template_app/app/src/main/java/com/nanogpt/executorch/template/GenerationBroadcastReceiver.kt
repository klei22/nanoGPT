package com.nanogpt.executorch.template

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.core.content.ContextCompat
import java.util.ArrayList

class GenerationBroadcastReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val prompt = intent.getStringExtra(EXTRA_PROMPT) ?: "Hello"
        val maxTokens = intent.getIntExtra(EXTRA_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        val contextLength = if (intent.hasExtra(EXTRA_CONTEXT_LENGTH)) {
            intent.getIntExtra(EXTRA_CONTEXT_LENGTH, DEFAULT_CONTEXT_LENGTH)
        } else {
            null
        }
        val sweepSpec = intent.getStringExtra(EXTRA_CONTEXT_SWEEP)
        val sweep = sweepSpec?.split(',')?.mapNotNull { it.trim().toIntOrNull() }

        Log.i(TAG, "Received generation request: prompt='${prompt}', maxTokens=${maxTokens}, contextLength=${contextLength}, sweep=${sweep}")

        val serviceIntent = Intent(context, NanoGptService::class.java).apply {
            putExtra(EXTRA_PROMPT, prompt)
            putExtra(EXTRA_MAX_TOKENS, maxTokens)
            contextLength?.let { putExtra(EXTRA_CONTEXT_LENGTH, it) }
            sweep?.let { putIntegerArrayListExtra(EXTRA_CONTEXT_SWEEP, ArrayList(it)) }
        }

        ContextCompat.startForegroundService(context, serviceIntent)
    }

    companion object {
        private const val TAG = "NanoGPTTemplate"

        const val EXTRA_PROMPT = "prompt"
        const val EXTRA_MAX_TOKENS = "max_tokens"
        const val EXTRA_CONTEXT_LENGTH = "context_length"
        const val EXTRA_CONTEXT_SWEEP = "context_sweep"

        private const val DEFAULT_MAX_TOKENS = 32
        private const val DEFAULT_CONTEXT_LENGTH = 128
    }
}
