package com.nanogpt.benchmark;

import android.app.Activity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    static { System.loadLibrary("nanogpt_benchmark"); }

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private TextView output;

    private native String runBenchmark(int promptTokens, int generateTokens, int threads);

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        output = new TextView(this);
        output.setTextSize(14f);
        output.setPadding(24, 24, 24, 24);
        output.setText("NanoGPT Android Benchmark\n\n" +
                "Convert a checkpoint to app/src/main/assets/nanogpt.bin, build, install, then tap Run.\n" +
                "This JNI path executes a simple fp32 CPU forward pass for repeatable hardware comparisons.\n");

        Button run = new Button(this);
        run.setText("Run 32-token benchmark");
        run.setOnClickListener(v -> startBenchmark());

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.addView(run, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        ScrollView scroller = new ScrollView(this);
        scroller.addView(output);
        root.addView(scroller, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);
    }

    private void startBenchmark() {
        output.append("\nRunning...\n");
        executor.submit(() -> {
            long start = System.nanoTime();
            String result = runBenchmark(16, 32, Runtime.getRuntime().availableProcessors());
            double wall = (System.nanoTime() - start) / 1_000_000_000.0;
            runOnUiThread(() -> output.append(String.format(Locale.US, "%s\nWall time: %.3fs\n", result, wall)));
        });
    }

    @Override protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }
}
