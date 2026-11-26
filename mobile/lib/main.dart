// Sidecar for Android.
//
// The app is a window onto the web app the sidecar serves at /app/ on your
// Mac, reached over Tailscale. Everything you see is that page; what this
// file adds is what a web page cannot do well on Android:
//
//   - appear in the Share menu of every app, and hand the shared text over
//   - voice input through the system recogniser (WebView has no Web Speech)
//   - read replies aloud through the system text-to-speech engine
//   - save and share generated images as real files
//   - open links in their own apps, and a proper offline screen
//
// The page talks to this side through a JavaScript channel named
// SidecarNative, and this side answers by calling window.__sidecar* hooks.
// The page works without any of it (in Chrome), so neither side depends on
// the other being there.
import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:gal/gal.dart';
import 'package:http/http.dart' as http;
import 'package:material_ui/material_ui.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:webview_flutter/webview_flutter.dart';

// A personal build can bake its Mac in:
//   flutter build apk --dart-define=SIDECAR_URL=https://my-mac.tailnet.ts.net:8443
// Without it the app asks on first launch.
const bakedUrl = String.fromEnvironment('SIDECAR_URL');

const indigo = Color(0xFF4F46E5);
const violet = Color(0xFF7C3AED);
const darkBg = Color(0xFF0D0E12);
const lightBg = Color(0xFFF6F6F9);
const intents = MethodChannel('sidecar/intents');

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  runApp(const SidecarApp());
}

class SidecarApp extends StatelessWidget {
  const SidecarApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Sidecar',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(colorSchemeSeed: indigo, brightness: Brightness.light, scaffoldBackgroundColor: lightBg),
        darkTheme: ThemeData(colorSchemeSeed: indigo, brightness: Brightness.dark, scaffoldBackgroundColor: darkBg),
        home: const Root(),
      );
}

/// Decides between the setup screen and the app, and routes incoming
/// intents (share, connect links) to whichever is showing.
class Root extends StatefulWidget {
  const Root({super.key});
  @override
  State<Root> createState() => _RootState();
}

class _RootState extends State<Root> {
  String? base;
  bool loaded = false;
  Map<String, String>? pendingShare;
  // Replaced whenever the address changes, so a new Mac gets a new WebView
  // rather than the old one's state.
  var shell = GlobalKey<_ShellState>();

  @override
  void initState() {
    super.initState();
    _boot();
    intents.setMethodCallHandler((call) async {
      if (call.method == 'intent') _handle(Map<String, String>.from(call.arguments as Map));
    });
  }

  Future<void> _boot() async {
    final prefs = await SharedPreferences.getInstance();
    base = prefs.getString('url') ?? (bakedUrl.isEmpty ? null : normalise(bakedUrl));
    final first = await intents.invokeMethod<Map>('initial');
    setState(() => loaded = true);
    if (first != null) _handle(Map<String, String>.from(first));
  }

  Future<void> _handle(Map<String, String> i) async {
    if (i['kind'] == 'connect' && (i['url'] ?? '').isNotEmpty) {
      await _setUrl(normalise(i['url']!));
      _toast('Connected to ${Uri.parse(base!).host.split('.').first}');
    } else if (i['kind'] == 'share') {
      if (shell.currentState != null) {
        shell.currentState!.receiveShare(i);
      } else {
        pendingShare = i;
      }
    }
  }

  Future<void> _setUrl(String? url) async {
    final prefs = await SharedPreferences.getInstance();
    if (url == null) {
      await prefs.remove('url');
    } else {
      await prefs.setString('url', url);
    }
    setState(() { base = url; shell = GlobalKey<_ShellState>(); });
  }

  void _toast(String text) =>
      ScaffoldMessenger.maybeOf(context)?.showSnackBar(SnackBar(content: Text(text), behavior: SnackBarBehavior.floating));

  @override
  Widget build(BuildContext context) {
    if (!loaded) return const Scaffold();
    if (base == null) return Setup(onConnect: _setUrl);
    final share = pendingShare;
    pendingShare = null;
    return Shell(key: shell, base: base!, initialShare: share, onReconfigure: () => _setUrl(null));
  }
}

/// https://mac.tail.ts.net:8443/app/ -> https://mac.tail.ts.net:8443
String normalise(String raw) {
  var s = raw.trim();
  if (!s.contains('://')) s = 'https://$s';
  final u = Uri.parse(s);
  return Uri(scheme: u.scheme, host: u.host, port: u.hasPort ? u.port : null).toString();
}

// ── setup ───────────────────────────────────────────────────────────────────

class Setup extends StatefulWidget {
  const Setup({super.key, required this.onConnect});
  final Future<void> Function(String) onConnect;
  @override
  State<Setup> createState() => _SetupState();
}

class _SetupState extends State<Setup> {
  final field = TextEditingController();
  String? error;
  bool checking = false;

  Future<void> _connect() async {
    final raw = field.text.trim();
    if (raw.isEmpty) return;
    final url = normalise(raw);
    setState(() { checking = true; error = null; });
    try {
      final r = await http.get(Uri.parse('$url/health')).timeout(const Duration(seconds: 8));
      if (r.statusCode != 200 || !r.body.contains('"ok"')) throw Exception('answered, but not like a sidecar (HTTP ${r.statusCode})');
      HapticFeedback.mediumImpact();
      await widget.onConnect(url);
    } catch (e) {
      setState(() => error = e is SocketException || e.toString().contains('Timeout')
          ? "Couldn't reach it. Is Tailscale on, on this phone, and is the Mac awake?"
          : e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => checking = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    final muted = t.colorScheme.onSurfaceVariant;
    return Scaffold(
      body: SafeArea(
        child: ListView(padding: const EdgeInsets.fromLTRB(24, 48, 24, 24), children: [
          Center(child: Container(
            width: 72, height: 72,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(22),
              gradient: const LinearGradient(colors: [indigo, violet], begin: Alignment.topLeft, end: Alignment.bottomRight),
              boxShadow: [BoxShadow(color: indigo.withValues(alpha: .35), blurRadius: 28, offset: const Offset(0, 10))],
            ),
            child: const _Mark(size: 40),
          )),
          const SizedBox(height: 24),
          Text('Connect to your Mac', textAlign: TextAlign.center, style: t.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 8),
          Text('Sidecar runs on your Mac. This app reaches it privately over Tailscale — at home or anywhere.',
              textAlign: TextAlign.center, style: t.textTheme.bodyMedium?.copyWith(color: muted)),
          const SizedBox(height: 28),
          _Step(n: 1, title: 'On the Mac, run', mono: 'llm-sidecar phone'),
          _Step(n: 2, title: 'Make sure Tailscale is on, on this phone'),
          _Step(n: 3, title: 'Paste the address it printed — or open it in Chrome and tap “Use in the Android app”'),
          const SizedBox(height: 20),
          TextField(
            controller: field,
            keyboardType: TextInputType.url,
            autocorrect: false,
            textInputAction: TextInputAction.go,
            onSubmitted: (_) => _connect(),
            decoration: InputDecoration(
              hintText: 'https://your-mac.tailnet.ts.net:8443',
              errorText: error, errorMaxLines: 3,
              filled: true,
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: BorderSide.none),
              suffixIcon: IconButton(
                icon: const Icon(Icons.content_paste_rounded),
                tooltip: 'Paste',
                onPressed: () async {
                  final c = await Clipboard.getData('text/plain');
                  if (c?.text != null) field.text = c!.text!.trim();
                },
              ),
            ),
          ),
          const SizedBox(height: 14),
          FilledButton(
            onPressed: checking ? null : _connect,
            style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14))),
            child: checking
                ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.4))
                : const Text('Connect', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
          ),
        ]),
      ),
    );
  }
}

class _Step extends StatelessWidget {
  const _Step({required this.n, required this.title, this.mono});
  final int n;
  final String title;
  final String? mono;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Container(
          width: 26, height: 26, alignment: Alignment.center,
          decoration: BoxDecoration(color: t.colorScheme.primaryContainer, shape: BoxShape.circle),
          child: Text('$n', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 13, color: t.colorScheme.onPrimaryContainer)),
        ),
        const SizedBox(width: 12),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Padding(padding: const EdgeInsets.only(top: 3), child: Text(title, style: t.textTheme.bodyMedium)),
          if (mono != null)
            GestureDetector(
              onTap: () { Clipboard.setData(ClipboardData(text: mono!)); HapticFeedback.selectionClick(); },
              child: Container(
                margin: const EdgeInsets.only(top: 6),
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(color: t.colorScheme.surfaceContainerHighest, borderRadius: BorderRadius.circular(8)),
                child: Text(mono!, style: const TextStyle(fontFamily: 'monospace', fontSize: 13)),
              ),
            ),
        ])),
      ]),
    );
  }
}

/// The two circles, drawn natively so the setup and offline screens match
/// the web app without loading anything.
class _Mark extends StatelessWidget {
  const _Mark({required this.size});
  final double size;
  @override
  Widget build(BuildContext context) => Center(
        child: SizedBox(width: size, height: size, child: CustomPaint(painter: _MarkPainter())),
      );
}

class _MarkPainter extends CustomPainter {
  @override
  void paint(Canvas c, Size s) {
    final u = s.width / 24;
    c.drawCircle(Offset(8.4 * u, 12 * u), 6.2 * u, Paint()..color = Colors.white);
    c.drawCircle(Offset(17.9 * u, 12 * u), 3.1 * u, Paint()..color = Colors.white.withValues(alpha: .76));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

// ── the app ─────────────────────────────────────────────────────────────────

class Shell extends StatefulWidget {
  const Shell({super.key, required this.base, this.initialShare, required this.onReconfigure});
  final String base;
  final Map<String, String>? initialShare;
  final VoidCallback onReconfigure;
  @override
  State<Shell> createState() => _ShellState();
}

class _ShellState extends State<Shell> with WidgetsBindingObserver {
  late final WebViewController web;
  late final String host = Uri.parse(widget.base).host;
  final speech = SpeechToText();
  final tts = FlutterTts();
  bool speechReady = false;
  int progress = 0;
  String? error;
  bool pageReady = false;
  final queue = <String>[];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    web = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..addJavaScriptChannel('SidecarNative', onMessageReceived: (m) => _fromPage(m.message))
      ..setNavigationDelegate(NavigationDelegate(
        onProgress: (p) => setState(() => progress = p),
        onPageStarted: (_) => setState(() { error = null; pageReady = false; }),
        onPageFinished: (_) async {
          pageReady = true;
          // Give the page's script a beat to define its hooks.
          await Future.delayed(const Duration(milliseconds: 250));
          for (final js in queue) { await web.runJavaScript(js); }
          queue.clear();
        },
        onWebResourceError: (e) {
          if (e.isForMainFrame ?? true) setState(() => error = e.description);
        },
        onNavigationRequest: (req) {
          final u = Uri.tryParse(req.url);
          if (u == null || u.host == host || u.scheme == 'about' || u.scheme == 'data' || u.scheme == 'blob') {
            return NavigationDecision.navigate;
          }
          if (u.scheme == 'llmsidecar') return NavigationDecision.prevent;
          // Sources, YouTube, maps, everything else: its own app.
          launchUrl(u, mode: LaunchMode.externalApplication);
          return NavigationDecision.prevent;
        },
      ));
    _start();
    tts.setCompletionHandler(() => _js('window.__sidecarSpeaking && window.__sidecarSpeaking(false)'));
  }

  Future<void> _start() async {
    final ua = await web.getUserAgent();
    await web.setUserAgent('${ua ?? ''} LLMSidecarApp/1');
    await web.loadRequest(Uri.parse('${widget.base}/app/'));
    if (widget.initialShare != null) receiveShare(widget.initialShare!);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    tts.stop();
    speech.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState s) {
    // Coming back after the Mac slept or Tailscale reconnected: retry the
    // offline screen automatically rather than waiting for a tap.
    if (s == AppLifecycleState.resumed && error != null) reload();
    if (s == AppLifecycleState.paused) { tts.stop(); speech.stop(); }
  }

  void reload() {
    setState(() => error = null);
    web.loadRequest(Uri.parse('${widget.base}/app/'));
  }

  /// Run JS now if the page is ready, else once it is.
  Future<void> _js(String code) async {
    if (pageReady) {
      try { await web.runJavaScript(code); } catch (_) {}
    } else {
      queue.add(code);
    }
  }

  void receiveShare(Map<String, String> i) {
    HapticFeedback.mediumImpact();
    final payload = jsonEncode({'title': i['title'] ?? '', 'text': i['text'] ?? '', 'url': ''});
    _js('window.__sidecarShare && window.__sidecarShare($payload)');
  }

  void _voice(Map<String, Object?> ev) => _js('window.__sidecarVoice && window.__sidecarVoice(${jsonEncode(ev)})');

  Future<void> _fromPage(String raw) async {
    final Map<String, dynamic> m;
    try { m = jsonDecode(raw) as Map<String, dynamic>; } catch (_) { return; }
    switch (m['type']) {
      case 'haptic':
        m['kind'] == 'heavy' ? HapticFeedback.mediumImpact() : HapticFeedback.selectionClick();
      case 'listen':
        await _listen();
      case 'stopListening':
        await speech.stop();
      case 'speak':
        await tts.stop();
        await tts.speak((m['text'] as String?) ?? '');
      case 'shareText':
        await SharePlus.instance.share(ShareParams(text: (m['text'] as String?) ?? ''));
      case 'shareImage':
        final f = await _download(m['url'] as String);
        if (f != null) await SharePlus.instance.share(ShareParams(files: [XFile(f.path, mimeType: 'image/png')], text: m['text'] as String?));
      case 'saveImage':
        await _save(m['url'] as String);
      case 'reconfigure':
        widget.onReconfigure();
    }
  }

  Future<void> _listen() async {
    if (!speechReady) {
      speechReady = await speech.initialize(
        onError: (e) => _voice({'error': e.errorMsg == 'error_no_match' ? "Didn't catch that" : e.errorMsg.replaceAll('error_', '').replaceAll('_', ' ')}),
        onStatus: (s) { if (s == 'done' || s == 'notListening') _voice({'listening': false}); },
      );
      if (!speechReady) {
        _voice({'error': 'Speech recognition is unavailable — check the microphone permission.'});
        return;
      }
    }
    await tts.stop();
    await speech.listen(
      onResult: (SpeechRecognitionResult r) => _voice({'text': r.recognizedWords, 'final': r.finalResult}),
      listenOptions: SpeechListenOptions(
        partialResults: true, cancelOnError: true,
        listenFor: const Duration(minutes: 1),
        // Long enough to think mid-sentence, short enough that stopping
        // talking feels like pressing send.
        pauseFor: const Duration(seconds: 3),
      ),
    );
  }

  Future<File?> _download(String url) async {
    try {
      final r = await http.get(Uri.parse(url));
      if (r.statusCode != 200) throw Exception('HTTP ${r.statusCode}');
      final dir = await getTemporaryDirectory();
      final name = Uri.parse(url).pathSegments.last;
      final f = File('${dir.path}/$name');
      await f.writeAsBytes(r.bodyBytes);
      return f;
    } catch (e) {
      _snack("Couldn't fetch the image: $e");
      return null;
    }
  }

  Future<void> _save(String url) async {
    try {
      final r = await http.get(Uri.parse(url));
      if (r.statusCode != 200) throw Exception('HTTP ${r.statusCode}');
      if (!await Gal.hasPermission()) await Gal.requestPermission();
      await Gal.putImageBytes(r.bodyBytes, name: Uri.parse(url).pathSegments.last.replaceAll('.png', ''));
      HapticFeedback.mediumImpact();
      _snack('Saved to your gallery');
    } catch (e) {
      _snack("Couldn't save: $e");
    }
  }

  void _snack(String text) => ScaffoldMessenger.of(context)
      .showSnackBar(SnackBar(content: Text(text), behavior: SnackBarBehavior.floating));

  @override
  Widget build(BuildContext context) {
    final dark = MediaQuery.platformBrightnessOf(context) == Brightness.dark;
    SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      systemNavigationBarColor: Colors.transparent,
      statusBarIconBrightness: dark ? Brightness.light : Brightness.dark,
      systemNavigationBarIconBrightness: dark ? Brightness.light : Brightness.dark,
    ));
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) return;
        // The page pushes a history entry for every sheet and the drawer,
        // so "back" closes those first, as it would in a native app.
        if (error == null && await web.canGoBack()) {
          web.goBack();
        } else {
          SystemNavigator.pop();
        }
      },
      child: Scaffold(
        backgroundColor: dark ? darkBg : lightBg,
        body: SafeArea(
          child: Stack(children: [
            WebViewWidget(controller: web),
            if (progress < 100 && error == null)
              LinearProgressIndicator(value: progress / 100, minHeight: 2, color: indigo, backgroundColor: Colors.transparent),
            if (error != null) _Offline(host: host, detail: error!, onRetry: reload, onChange: widget.onReconfigure),
          ]),
        ),
      ),
    );
  }
}

class _Offline extends StatelessWidget {
  const _Offline({required this.host, required this.detail, required this.onRetry, required this.onChange});
  final String host;
  final String detail;
  final VoidCallback onRetry;
  final VoidCallback onChange;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context);
    return Container(
      color: t.scaffoldBackgroundColor,
      padding: const EdgeInsets.all(28),
      alignment: Alignment.center,
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Container(
          width: 72, height: 72,
          decoration: BoxDecoration(color: t.colorScheme.errorContainer, borderRadius: BorderRadius.circular(22)),
          child: Icon(Icons.wifi_off_rounded, size: 34, color: t.colorScheme.onErrorContainer),
        ),
        const SizedBox(height: 20),
        Text("Can't reach your Mac", style: t.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
        const SizedBox(height: 10),
        Text(
          'Check that Tailscale is connected on this phone, and that the Mac is awake with the sidecar running.',
          textAlign: TextAlign.center,
          style: t.textTheme.bodyMedium?.copyWith(color: t.colorScheme.onSurfaceVariant),
        ),
        const SizedBox(height: 24),
        FilledButton.icon(
          onPressed: onRetry,
          icon: const Icon(Icons.refresh_rounded),
          label: const Text('Try again'),
          style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 26, vertical: 14)),
        ),
        TextButton(onPressed: onChange, child: const Text('Connect to a different Mac')),
        const SizedBox(height: 10),
        Text(host, style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.outline, fontFamily: 'monospace')),
        Text(detail, textAlign: TextAlign.center, style: t.textTheme.bodySmall?.copyWith(color: t.colorScheme.outline)),
      ]),
    );
  }
}
