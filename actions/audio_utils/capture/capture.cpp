// capture.cpp — capteur audio WASAPI autonome (Windows), pour utils.
//
// But : permettre à la TUI utils (qui tourne en WSL) de capturer, côté Windows,
// plusieurs endpoints audio simultanément — micro(s) ET sortie(s) système — via
// le loopback WASAPI, qui est une API Windows INTÉGRÉE : aucun driver, aucun
// filtre DirectShow, aucun regsvr32, aucun VC++ redist, aucun admin requis.
//
// Le binaire est volontairement minimal et n'écrit QUE du PCM ; tout l'encodage
// FLAC et la transcription restent côté WSL (ffmpeg + transcribe_audio.py).
//
// ── Interface CLI (contrat consommé par recorder.py) ─────────────────────────
//
//   capture.exe --list
//       → écrit sur stdout un tableau JSON des endpoints :
//         [{"id":"...","name":"...","kind":"input|output","channels":N}, ...]
//         render endpoints  → kind "output" (capturés en loopback)
//         capture endpoints → kind "input"
//
//   capture.exe --rate 48000 --source <id> [--source <id> ...]
//       → capture les endpoints donnés (ordre = ordre des --source) et écrit sur
//         stdout un flux PCM s16le INTERLEAVÉ : pour chaque trame, les canaux de
//         la source 0, puis ceux de la source 1, etc. (nb total de canaux = somme
//         des canaux de chaque source). Cadence à `rate` Hz.
//       → écrit sur stderr, ~10×/s, des lignes "LEVEL <idx> <rms 0..1>" par source
//         (pour les vumètres de la TUI).
//       → s'arrête quand stdout est fermé (pipe cassé), sur Ctrl-C, ou à la mort
//         du process (terminate côté WSL).
//
// Build : voir build.sh (cross-compile mingw-w64) / build.bat (MSVC ou g++).

#define INITGUID
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <mmreg.h>
#include <io.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>

static std::atomic<bool> g_stop{false};

static BOOL WINAPI ctrl_handler(DWORD) { g_stop = true; return TRUE; }

// ── utilitaires ──────────────────────────────────────────────────────────────

static std::string wide_to_utf8(const wchar_t* w) {
    if (!w) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return std::string();
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}

static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w(n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

static std::string json_escape(const std::string& in) {
    std::string out;
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out += b; }
                else out += c;
        }
    }
    return out;
}

static std::string endpoint_name(IMMDevice* dev) {
    std::string name;
    IPropertyStore* props = nullptr;
    if (SUCCEEDED(dev->OpenPropertyStore(STGM_READ, &props)) && props) {
        PROPVARIANT pv; PropVariantInit(&pv);
        if (SUCCEEDED(props->GetValue(PKEY_Device_FriendlyName, &pv)) && pv.vt == VT_LPWSTR)
            name = wide_to_utf8(pv.pwszVal);
        PropVariantClear(&pv);
        props->Release();
    }
    return name;
}

// ── énumération (--list) ─────────────────────────────────────────────────────

static int do_list() {
    IMMDeviceEnumerator* en = nullptr;
    if (FAILED(CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL,
                                IID_IMMDeviceEnumerator, (void**)&en)))
        return 1;

    fputs("[", stdout);
    bool first = true;
    const EDataFlow flows[2] = {eCapture, eRender};
    const char* kinds[2]     = {"input", "output"};

    for (int fi = 0; fi < 2; ++fi) {
        IMMDeviceCollection* col = nullptr;
        if (FAILED(en->EnumAudioEndpoints(flows[fi], DEVICE_STATE_ACTIVE, &col)) || !col)
            continue;
        UINT count = 0; col->GetCount(&count);
        for (UINT i = 0; i < count; ++i) {
            IMMDevice* dev = nullptr;
            if (FAILED(col->Item(i, &dev)) || !dev) continue;
            LPWSTR idw = nullptr;
            int channels = 2;
            if (SUCCEEDED(dev->GetId(&idw)) && idw) {
                IAudioClient* ac = nullptr;
                if (SUCCEEDED(dev->Activate(IID_IAudioClient, CLSCTX_ALL, nullptr, (void**)&ac)) && ac) {
                    WAVEFORMATEX* mix = nullptr;
                    if (SUCCEEDED(ac->GetMixFormat(&mix)) && mix) {
                        channels = mix->nChannels;
                        CoTaskMemFree(mix);
                    }
                    ac->Release();
                }
                std::string id   = wide_to_utf8(idw);
                std::string name = endpoint_name(dev);
                if (name.empty()) name = id;
                if (!first) fputs(",", stdout);
                first = false;
                printf("{\"id\":\"%s\",\"name\":\"%s\",\"kind\":\"%s\",\"channels\":%d}",
                       json_escape(id).c_str(), json_escape(name).c_str(),
                       kinds[fi], channels);
                CoTaskMemFree(idw);
            }
            dev->Release();
        }
        col->Release();
    }
    fputs("]\n", stdout);
    en->Release();
    return 0;
}

// ── capture multi-sources ────────────────────────────────────────────────────

struct SourceCtx {
    std::string       id;
    IAudioClient*     client  = nullptr;
    IAudioCaptureClient* cap  = nullptr;
    int               channels = 2;
    bool              loopback = false;
    std::mutex        mtx;
    std::deque<int16_t> buf;          // PCM s16 interleavé, à `rate` Hz, `channels`
    std::thread       th;
};

// Convertit un paquet WASAPI (float32 ou int16, déjà à `rate` grâce à
// AUTOCONVERTPCM) en int16 et l'empile dans le buffer de la source.
static void push_packet(SourceCtx& s, const BYTE* data, UINT32 frames,
                        const WAVEFORMATEX* fmt, bool silent) {
    const int ch = s.channels;
    std::vector<int16_t> out((size_t)frames * ch);
    if (silent || !data) {
        std::fill(out.begin(), out.end(), (int16_t)0);
    } else if (fmt->wBitsPerSample == 16) {
        memcpy(out.data(), data, (size_t)frames * ch * sizeof(int16_t));
    } else if (fmt->wBitsPerSample == 32) {
        // float32 [-1,1] → s16
        const float* f = reinterpret_cast<const float*>(data);
        for (size_t i = 0; i < (size_t)frames * ch; ++i) {
            float v = f[i];
            if (v > 1.0f) v = 1.0f; else if (v < -1.0f) v = -1.0f;
            out[i] = (int16_t)lrintf(v * 32767.0f);
        }
    } else {
        std::fill(out.begin(), out.end(), (int16_t)0);
    }
    std::lock_guard<std::mutex> lk(s.mtx);
    s.buf.insert(s.buf.end(), out.begin(), out.end());
}

static void capture_thread(SourceCtx* s, WAVEFORMATEX* fmt) {
    s->client->Start();
    while (!g_stop) {
        UINT32 packet = 0;
        if (FAILED(s->cap->GetNextPacketSize(&packet))) break;
        if (packet == 0) { Sleep(5); continue; }
        BYTE* data = nullptr; UINT32 frames = 0; DWORD flags = 0;
        if (FAILED(s->cap->GetBuffer(&data, &frames, &flags, nullptr, nullptr))) break;
        push_packet(*s, data, frames, fmt, (flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0);
        s->cap->ReleaseBuffer(frames);
    }
    s->client->Stop();
    CoTaskMemFree(fmt);
}

static IMMDevice* get_device(IMMDeviceEnumerator* en, const std::string& id, bool& is_render) {
    std::wstring idw = utf8_to_wide(id);
    IMMDevice* dev = nullptr;
    if (FAILED(en->GetDevice(idw.c_str(), &dev)) || !dev) return nullptr;
    is_render = false;
    IMMEndpoint* ep = nullptr;
    if (SUCCEEDED(dev->QueryInterface(IID_IMMEndpoint, (void**)&ep)) && ep) {
        EDataFlow flow = eRender;
        if (SUCCEEDED(ep->GetDataFlow(&flow))) is_render = (flow == eRender);
        ep->Release();
    }
    return dev;
}

static int do_capture(int rate, const std::vector<std::string>& ids) {
    IMMDeviceEnumerator* en = nullptr;
    if (FAILED(CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL,
                                IID_IMMDeviceEnumerator, (void**)&en)))
        return 1;

    std::vector<SourceCtx*> srcs;
    for (const auto& id : ids) {
        bool is_render = false;
        IMMDevice* dev = get_device(en, id, is_render);
        if (!dev) { fprintf(stderr, "ERR source introuvable: %s\n", id.c_str()); continue; }

        IAudioClient* client = nullptr;
        if (FAILED(dev->Activate(IID_IAudioClient, CLSCTX_ALL, nullptr, (void**)&client)) || !client) {
            dev->Release(); continue;
        }
        WAVEFORMATEX* mix = nullptr;
        client->GetMixFormat(&mix);
        int channels = mix ? mix->nChannels : 2;

        // Format demandé : PCM s16, `rate` Hz, canaux du mix. AUTOCONVERTPCM
        // laisse le mixeur partagé faire la conversion de fréquence/format.
        WAVEFORMATEX want;
        memset(&want, 0, sizeof want);
        want.wFormatTag      = WAVE_FORMAT_PCM;
        want.nChannels       = (WORD)channels;
        want.nSamplesPerSec  = (DWORD)rate;
        want.wBitsPerSample  = 16;
        want.nBlockAlign     = (WORD)(channels * 2);
        want.nAvgBytesPerSec = (DWORD)rate * want.nBlockAlign;
        want.cbSize          = 0;
        if (mix) CoTaskMemFree(mix);

        DWORD flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
                    | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
        if (is_render) flags |= AUDCLNT_STREAMFLAGS_LOOPBACK;

        const REFERENCE_TIME dur = 2000000; // 200 ms
        HRESULT hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, dur, 0, &want, nullptr);
        if (FAILED(hr)) {
            fprintf(stderr, "ERR Initialize 0x%08lx pour %s\n", (unsigned long)hr, id.c_str());
            client->Release(); dev->Release(); continue;
        }
        IAudioCaptureClient* cap = nullptr;
        if (FAILED(client->GetService(IID_IAudioCaptureClient, (void**)&cap)) || !cap) {
            client->Release(); dev->Release(); continue;
        }

        SourceCtx* s = new SourceCtx();
        s->id = id; s->client = client; s->cap = cap;
        s->channels = channels; s->loopback = is_render;
        srcs.push_back(s);

        WAVEFORMATEX* fcopy = (WAVEFORMATEX*)CoTaskMemAlloc(sizeof(WAVEFORMATEX));
        *fcopy = want;
        s->th = std::thread(capture_thread, s, fcopy);
        dev->Release();
    }

    if (srcs.empty()) { en->Release(); return 1; }

    // stdout en binaire (pas de traduction CRLF)
    _setmode(_fileno(stdout), _O_BINARY);

    const int chunkFrames = rate / 100;          // 10 ms
    const double tickSec   = 0.01;
    std::vector<std::vector<int16_t>> pull(srcs.size());

    LARGE_INTEGER freq, t0; QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&t0);
    long long tick = 0;
    int level_div = 0;

    while (!g_stop) {
        // pacing : attendre le prochain tick 10 ms
        ++tick;
        double target = tick * tickSec;
        for (;;) {
            LARGE_INTEGER now; QueryPerformanceCounter(&now);
            double el = double(now.QuadPart - t0.QuadPart) / double(freq.QuadPart);
            if (el >= target || g_stop) break;
            Sleep(1);
        }

        // tirer chunkFrames de chaque source (zero-fill si sous-alimenté)
        for (size_t i = 0; i < srcs.size(); ++i) {
            SourceCtx* s = srcs[i];
            const size_t need = (size_t)chunkFrames * s->channels;
            pull[i].assign(need, 0);
            std::lock_guard<std::mutex> lk(s->mtx);
            size_t take = need < s->buf.size() ? need : s->buf.size();
            for (size_t k = 0; k < take; ++k) pull[i][k] = s->buf[k];
            s->buf.erase(s->buf.begin(), s->buf.begin() + take);
        }

        // interleave inter-sources, trame par trame → un seul flux PCM
        std::vector<int16_t> out;
        out.reserve((size_t)chunkFrames * 4);
        for (int f = 0; f < chunkFrames; ++f) {
            for (size_t i = 0; i < srcs.size(); ++i) {
                const int ch = srcs[i]->channels;
                const size_t base = (size_t)f * ch;
                for (int c = 0; c < ch; ++c) out.push_back(pull[i][base + c]);
            }
        }
        if (!out.empty()) {
            size_t w = fwrite(out.data(), sizeof(int16_t), out.size(), stdout);
            if (w < out.size()) break;          // pipe fermé côté WSL → stop
            fflush(stdout);
        }

        // niveaux (~10×/s) : RMS par source sur le chunk
        if (++level_div >= 1) {
            level_div = 0;
            for (size_t i = 0; i < srcs.size(); ++i) {
                const int ch = srcs[i]->channels;
                double acc = 0; size_t n = (size_t)chunkFrames * ch;
                for (size_t k = 0; k < n && k < pull[i].size(); ++k) {
                    double v = pull[i][k] / 32768.0; acc += v * v;
                }
                double rms = n ? std::sqrt(acc / n) : 0.0;
                fprintf(stderr, "LEVEL %zu %.4f\n", i, rms);
            }
            fflush(stderr);
        }
    }

    g_stop = true;
    for (auto* s : srcs) { if (s->th.joinable()) s->th.join(); }
    for (auto* s : srcs) {
        if (s->cap) s->cap->Release();
        if (s->client) s->client->Release();
        delete s;
    }
    en->Release();
    return 0;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SetConsoleCtrlHandler(ctrl_handler, TRUE);
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) {
        fprintf(stderr, "ERR CoInitializeEx\n");
        return 2;
    }

    bool list = false;
    int rate = 48000;
    std::vector<std::string> ids;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--list") list = true;
        else if (a == "--rate" && i + 1 < argc) rate = atoi(argv[++i]);
        else if (a == "--source" && i + 1 < argc) ids.push_back(argv[++i]);
    }

    int rc;
    if (list)            rc = do_list();
    else if (!ids.empty()) rc = do_capture(rate, ids);
    else { fprintf(stderr, "usage: capture.exe --list | --rate R --source ID [--source ID ...]\n"); rc = 2; }

    CoUninitialize();
    return rc;
}
