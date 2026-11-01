#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr uint32_t kMagic = 0x4e475054; // NGPT
constexpr uint32_t kVersion = 1;

struct Tensor { std::vector<float> v; };
struct Layer {
    Tensor ln1_g, ln2_g, q_w, k_w, v_w, proj_w, fc_w, fc_b, mlp_w;
};
struct Model {
    int block_size{}, vocab_size{}, n_layer{}, n_head{}, n_embd{}, hidden{};
    Tensor wte, wpe, lnf_g, lm_head;
    std::vector<Layer> layers;
};

static int32_t read_i32(const std::vector<uint8_t>& b, size_t& p) {
    if (p + 4 > b.size()) throw std::runtime_error("truncated int32");
    int32_t x; std::memcpy(&x, b.data() + p, 4); p += 4; return x;
}
static uint32_t read_u32(const std::vector<uint8_t>& b, size_t& p) { return static_cast<uint32_t>(read_i32(b, p)); }
static Tensor read_tensor(const std::vector<uint8_t>& b, size_t& p) {
    int32_t n = read_i32(b, p);
    if (n < 0 || p + static_cast<size_t>(n) * 4 > b.size()) throw std::runtime_error("truncated tensor");
    Tensor t; t.v.resize(n);
    std::memcpy(t.v.data(), b.data() + p, static_cast<size_t>(n) * 4); p += static_cast<size_t>(n) * 4;
    return t;
}

static std::vector<uint8_t> read_asset(JNIEnv* env, jobject assetManager, const char* name) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) throw std::runtime_error("missing AssetManager");
    AAsset* asset = AAssetManager_open(mgr, name, AASSET_MODE_BUFFER);
    if (!asset) throw std::runtime_error("missing asset app/src/main/assets/nanogpt.bin");
    off_t len = AAsset_getLength(asset);
    std::vector<uint8_t> bytes(static_cast<size_t>(len));
    int read = AAsset_read(asset, bytes.data(), bytes.size());
    AAsset_close(asset);
    if (read != len) throw std::runtime_error("could not read nanogpt.bin");
    return bytes;
}

static Model load_model(JNIEnv* env, jobject activity) {
    jclass cls = env->GetObjectClass(activity);
    jmethodID mid = env->GetMethodID(cls, "getAssets", "()Landroid/content/res/AssetManager;");
    jobject assets = env->CallObjectMethod(activity, mid);
    auto bytes = read_asset(env, assets, "nanogpt.bin");
    size_t p = 0;
    if (read_u32(bytes, p) != kMagic || read_u32(bytes, p) != kVersion) throw std::runtime_error("bad nanogpt.bin header");
    Model m;
    m.block_size = read_i32(bytes, p); m.vocab_size = read_i32(bytes, p); m.n_layer = read_i32(bytes, p);
    m.n_head = read_i32(bytes, p); m.n_embd = read_i32(bytes, p); m.hidden = read_i32(bytes, p);
    m.wte = read_tensor(bytes, p); m.wpe = read_tensor(bytes, p);
    m.layers.resize(m.n_layer);
    for (auto& l : m.layers) {
        l.ln1_g = read_tensor(bytes, p); l.ln2_g = read_tensor(bytes, p);
        l.q_w = read_tensor(bytes, p); l.k_w = read_tensor(bytes, p); l.v_w = read_tensor(bytes, p); l.proj_w = read_tensor(bytes, p);
        l.fc_w = read_tensor(bytes, p); l.fc_b = read_tensor(bytes, p); l.mlp_w = read_tensor(bytes, p);
    }
    m.lnf_g = read_tensor(bytes, p); m.lm_head = read_tensor(bytes, p);
    return m;
}

static void rmsnorm(const float* x, const float* gain, float* y, int n) {
    float ss = 0.f; for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float scale = 1.f / std::sqrt(ss / n + 1e-5f);
    for (int i = 0; i < n; ++i) y[i] = x[i] * scale * gain[i];
}
static void linear(const float* x, const float* w, float* y, int in, int out) {
    for (int o = 0; o < out; ++o) {
        const float* row = w + static_cast<size_t>(o) * in;
        float s = 0.f; for (int i = 0; i < in; ++i) s += row[i] * x[i];
        y[o] = s;
    }
}
static float gelu(float x) { return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }

static int forward_next(const Model& m, const std::vector<int>& ids, std::vector<float>& x) {
    const int T = std::min<int>(ids.size(), m.block_size), C = m.n_embd, H = m.n_head, HS = C / H;
    x.assign(static_cast<size_t>(T) * C, 0.f);
    int off = static_cast<int>(ids.size()) - T;
    for (int t = 0; t < T; ++t) {
        int id = ids[off + t] % m.vocab_size;
        for (int c = 0; c < C; ++c) x[static_cast<size_t>(t) * C + c] = m.wte.v[static_cast<size_t>(id) * C + c] + m.wpe.v[static_cast<size_t>(t) * C + c];
    }
    std::vector<float> norm(C), q(static_cast<size_t>(T)*C), k(static_cast<size_t>(T)*C), v(static_cast<size_t>(T)*C), att(static_cast<size_t>(T)*C), tmp(std::max(m.hidden, C));
    for (const auto& l : m.layers) {
        std::fill(att.begin(), att.end(), 0.f);
        for (int t = 0; t < T; ++t) {
            rmsnorm(&x[static_cast<size_t>(t)*C], l.ln1_g.v.data(), norm.data(), C);
            linear(norm.data(), l.q_w.v.data(), &q[static_cast<size_t>(t)*C], C, C);
            linear(norm.data(), l.k_w.v.data(), &k[static_cast<size_t>(t)*C], C, C);
            linear(norm.data(), l.v_w.v.data(), &v[static_cast<size_t>(t)*C], C, C);
        }
        std::vector<float> scores(T);
        for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
            float maxv = -1e30f, sum = 0.f;
            for (int s = 0; s <= t; ++s) {
                float dot = 0.f;
                for (int d = 0; d < HS; ++d) dot += q[(static_cast<size_t>(t)*C) + h*HS+d] * k[(static_cast<size_t>(s)*C) + h*HS+d];
                scores[s] = dot / std::sqrt(static_cast<float>(HS)); maxv = std::max(maxv, scores[s]);
            }
            for (int s = 0; s <= t; ++s) { scores[s] = std::exp(scores[s] - maxv); sum += scores[s]; }
            for (int d = 0; d < HS; ++d) {
                float val = 0.f; for (int s = 0; s <= t; ++s) val += (scores[s] / sum) * v[(static_cast<size_t>(s)*C) + h*HS+d];
                att[(static_cast<size_t>(t)*C) + h*HS+d] = val;
            }
        }
        for (int t = 0; t < T; ++t) { linear(&att[static_cast<size_t>(t)*C], l.proj_w.v.data(), tmp.data(), C, C); for (int c = 0; c < C; ++c) x[static_cast<size_t>(t)*C+c] += tmp[c]; }
        for (int t = 0; t < T; ++t) {
            rmsnorm(&x[static_cast<size_t>(t)*C], l.ln2_g.v.data(), norm.data(), C);
            linear(norm.data(), l.fc_w.v.data(), tmp.data(), C, m.hidden);
            for (int i = 0; i < m.hidden; ++i) tmp[i] = gelu(tmp[i] + l.fc_b.v[i]);
            std::vector<float> out(C); linear(tmp.data(), l.mlp_w.v.data(), out.data(), m.hidden, C);
            for (int c = 0; c < C; ++c) x[static_cast<size_t>(t)*C+c] += out[c];
        }
    }
    rmsnorm(&x[static_cast<size_t>(T-1)*C], m.lnf_g.v.data(), norm.data(), C);
    int best = 0; float bestv = -1e30f;
    for (int tok = 0; tok < m.vocab_size; ++tok) {
        float s = 0.f; const float* row = m.lm_head.v.data() + static_cast<size_t>(tok) * C;
        for (int c = 0; c < C; ++c) s += row[c] * norm[c];
        if (s > bestv) { bestv = s; best = tok; }
    }
    return best;
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nanogpt_benchmark_MainActivity_runBenchmark(JNIEnv* env, jobject thiz, jint promptTokens, jint generateTokens, jint threads) {
    try {
        auto t0 = std::chrono::steady_clock::now();
        Model model = load_model(env, thiz);
        std::vector<int> ids(promptTokens);
        for (int i = 0; i < promptTokens; ++i) ids[i] = (i * 1103515245u + 12345u) % model.vocab_size;
        std::vector<float> scratch;
        for (int i = 0; i < generateTokens; ++i) ids.push_back(forward_next(model, ids, scratch));
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double toks = generateTokens / sec;
        std::ostringstream os;
        os << "model=" << model.n_layer << "L/" << model.n_head << "H/" << model.n_embd << "C vocab=" << model.vocab_size
           << "\ngenerated=" << generateTokens << " threads_hint=" << threads
           << "\ntime=" << sec << "s tokens_per_second=" << toks << "\nlast_token=" << ids.back();
        return env->NewStringUTF(os.str().c_str());
    } catch (const std::exception& e) {
        std::string msg = std::string("Benchmark failed: ") + e.what();
        return env->NewStringUTF(msg.c_str());
    }
}
