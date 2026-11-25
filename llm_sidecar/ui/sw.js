// Service worker for the phone app.
//
// It keeps a copy of the app's own shell — the page, manifest and icons — and
// nothing else. That is what lets the installed app open when the Mac is
// asleep or Tailscale is off and say so, instead of showing Chrome's error
// page. Conversations already live on the phone, so they are still there.
//
// The shell is fetched network-first, so a new version of the sidecar reaches
// the phone the next time it is opened while connected. Every API call goes
// straight to the network and is never cached: a cached answer from a model
// would be a wrong answer presented as a current one.

const SHELL = "sidecar-shell-v1";
const FILES = ["/app/", "/app/manifest.webmanifest", "/app/icon-192.png", "/app/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== SHELL) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  const shell = url.pathname === "/app/" || FILES.includes(url.pathname);
  if (!shell) return;
  e.respondWith((async () => {
    // A share arrives as /app/?text=…; it is the same page.
    const key = url.pathname === "/app/" ? "/app/" : url.pathname;
    try {
      const fresh = await fetch(e.request);
      if (fresh.ok) (await caches.open(SHELL)).put(key, fresh.clone());
      return fresh;
    } catch {
      const hit = await caches.match(key);
      if (hit) return hit;
      throw new Error("offline");
    }
  })());
});
