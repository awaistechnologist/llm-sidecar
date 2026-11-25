# Sidecar for Android

Your Mac's AI from your phone — its local Ollama models, the Claude
subscription signed in on it, and OpenRouter — at home or anywhere, over
Tailscale. Nothing is exposed to your Wi-Fi or the internet.

The app is a window onto the web app the sidecar already serves at `/app/`.
You do not need it: Chrome's **Install app** gives you almost the same thing.
What the app adds is what a web page cannot do well on Android:

| | Chrome (installed) | This app |
|---|---|---|
| Chat, web answers with sources, images | ✓ | ✓ |
| In every app's **Share** menu | ✓ | ✓ |
| Voice input | ✓ (Chrome's recogniser) | ✓ (the system's, works offline-first) |
| Read replies aloud | ✓ | ✓ (system text-to-speech) |
| Save an image to the gallery | download | ✓ straight to Photos |
| Offline screen when the Mac is unreachable | ✓ | ✓, retries when you come back |
| Links open in their own apps | — | ✓ |

## Set up

On the Mac:

```bash
llm-sidecar service install   # keep the sidecar running
llm-sidecar phone             # put it on your tailnet; prints the address
```

On the phone: install Tailscale from the Play Store and sign in with the same
account. Then either open the printed address in Chrome and tap **Use in the
Android app**, or paste the address on the app's first screen.

## Build and install

With the phone plugged in and USB debugging on:

```bash
cd mobile
flutter build apk --release --target-platform android-arm64
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

To skip the setup screen on your own build, bake your Mac's address in. It is
compiled into that APK only and never written to the repository:

```bash
flutter build apk --release --target-platform android-arm64 \
  --dart-define=SIDECAR_URL=https://your-mac.your-tailnet.ts.net:8443
```

## How it fits together

```
phone ──Tailscale (WireGuard, your devices only)──▶ Mac :8443 (tailscale serve, HTTPS)
                                                       │
                                                       ▼
                                          llm-sidecar on 127.0.0.1:4001
                                     /app/  ·  /v1/chat  ·  /v1/answer  ·  images
```

The page and this app talk through a JavaScript channel, `SidecarNative`
(the page asks for voice, speech, haptics, sharing, saving), and the app
answers by calling `window.__sidecarVoice`, `window.__sidecarShare`. The page
checks for the channel and falls back to the web equivalent, so neither side
depends on the other.

Conversations are stored on the phone, not the Mac. Generated images are
stored on the Mac, in the sidecar's data directory, and shown from there.

Re-draw the launcher icons with `python scripts/make-icons.py` from the repo
root.
