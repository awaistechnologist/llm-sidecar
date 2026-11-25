package io.github.awaistechnologist.sidecar_app

import android.content.Intent
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

// Hands two kinds of incoming intent to the Flutter side:
//   - text shared from another app (Share → Sidecar)
//   - an llmsidecar://connect?url=... link, which sets the Mac's address
// The first one to arrive before Flutter is ready is held until it asks.
class MainActivity : FlutterActivity() {
    private var channel: MethodChannel? = null
    private var pending: Map<String, String>? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        pending = read(intent)
        channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "sidecar/intents").apply {
            setMethodCallHandler { call, result ->
                if (call.method == "initial") { result.success(pending); pending = null }
                else result.notImplemented()
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        read(intent)?.let { channel?.invokeMethod("intent", it) }
    }

    private fun read(i: Intent?): Map<String, String>? {
        if (i == null) return null
        if (i.action == Intent.ACTION_SEND && i.type == "text/plain") {
            return mapOf(
                "kind" to "share",
                "title" to (i.getStringExtra(Intent.EXTRA_SUBJECT) ?: ""),
                "text" to (i.getStringExtra(Intent.EXTRA_TEXT) ?: ""),
            )
        }
        val data = i.data
        if (i.action == Intent.ACTION_VIEW && data != null && data.scheme == "llmsidecar") {
            return mapOf("kind" to "connect", "url" to (data.getQueryParameter("url") ?: ""))
        }
        return null
    }
}
