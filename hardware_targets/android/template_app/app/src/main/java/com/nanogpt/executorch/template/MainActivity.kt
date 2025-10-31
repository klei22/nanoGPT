package com.nanogpt.executorch.template

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.nanogpt.executorch.template.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.promptText.text = getString(R.string.app_name)
        binding.statusText.text = getString(
            R.string.status_instructions,
            GenerationBroadcastReceiver.EXTRA_PROMPT,
            GenerationBroadcastReceiver.EXTRA_MAX_TOKENS,
            GenerationBroadcastReceiver.EXTRA_CONTEXT_LENGTH
        )
    }
}
