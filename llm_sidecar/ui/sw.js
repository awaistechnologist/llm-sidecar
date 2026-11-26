// Service worker for the phone app.
//
// It keeps a copy of the app's own shell — the page, manifest and icons — and
// nothing else. That is what lets the installed app open when the Mac is
// asleep or Tailscale is off and say so, instead of showing Chrome's error
// page. Conversations already live on the phone, so they are still there.
//
// The shell is served from the copy straight away and refreshed behind it
// (stale-while-revalidate). It used to be network-first, which made every
// open wait for a full download over Tailscale before anything appeared.
// When the refreshed page differs, the page is told and decides whether to
// switch at once (see "updated" in app.html). Every API call goes
// straight to the network and is never cached: a cached answer from a model
// would be a wrong answer presented as a current one.

const SHELL = "sidecar-shell-v2";
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
  // A share arrives as /app/?text=…; it is the same page.
  const key = url.pathname === "/app/" ? "/app/" : url.pathname;
  const refresh = (async () => {
    const cache = await caches.open(SHELL);
    const old = await cache.match(key);
    const fresh = await fetch(key, { cache: "no-cache" });
    if (!fresh.ok) return fresh;
    await cache.put(key, fresh.clone());
    if (old && key === "/app/" && (await old.text()) !== (await fresh.clone().text())) {
      const id = e.resultingClientId || e.clientId;
      const client = id && (await self.clients.get(id));
      (client ? [client] : await self.clients.matchAll()).forEach(
        (c) => c.postMessage({ type: "updated" }));
    }
    return fresh;
  })();
  e.respondWith((async () => {
    const hit = await caches.match(key);
    if (hit) {
      e.waitUntil(refresh.catch(() => {}));
      return hit;
    }
    return refresh;   // first visit: nothing cached yet
  })());
});
