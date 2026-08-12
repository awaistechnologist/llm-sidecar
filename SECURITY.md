# Security

## Reporting

Please report vulnerabilities privately through GitHub's
[security advisories](https://github.com/awaistechnologist/llm-sidecar/security/advisories/new)
rather than a public issue.

## What this software does with your credentials

- The OpenRouter API key is held **in memory** by the running process. It is
  written to disk only if you explicitly ask — `config.save(include_api_key=True)`,
  or ticking "remember on this machine" in the dashboard — and then it is
  plaintext JSON in your home directory.
- No endpoint returns the key. `/config` and `/status` report a masked
  preview (`…6ff9`) and nothing more.
- Nothing is sent anywhere except the model provider you configured, the
  search provider in use, and pages you ask it to read.

## Threat model, stated plainly

**The daemon binds `127.0.0.1` and is unauthenticated by default.** Anything
able to make loopback HTTP requests on your machine can therefore use it —
which means spending your OpenRouter credit and reading pages through it. That
is an acceptable trade for a single-user developer tool and an unacceptable
one for a shared or multi-tenant host.

If the machine is shared, set `LLM_SIDECAR_TOKEN` to require a bearer token.
Do not bind it to `0.0.0.0`.

The optional SearXNG container is likewise loopback-only, with its bot limiter
disabled — exposing it would hand anyone a free search proxy on your address.

## Scope

In scope: credential leakage, the daemon or dashboard doing something the
documentation says it doesn't, and prompt content reaching an unintended
provider.

Out of scope: a model returning a wrong answer. Verification is bounded by
what retrieval finds, and the README says so.
