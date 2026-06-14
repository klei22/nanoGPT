plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.nanogpt.benchmark"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.nanogpt.benchmark"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-mobile:1.20.0")
}
