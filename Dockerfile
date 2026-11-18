# For MCP directory introspection — not a way to run llm-sidecar for real.
#
# Glama builds this to start the server and ask it for its tool list, which
# is all its checks require. That works in a container: the MCP server lists
# its 17 tools without touching a model.
#
# Actually *using* it this way would not work, and the image does not pretend
# otherwise. The whole point of this project is reaching things that live on
# your machine — your Ollama, your files, your GPU, your key — and a
# container has none of them. Install it on the host instead:
#
#   uvx --from llm-sidecar llm-sidecar-mcp
#   pipx install llm-sidecar && llm-sidecar-mcp
#
# The [images] extra is deliberately absent: mflux is Apple-Silicon-only and
# pulls several GB, so it cannot build here and would be useless if it could.
FROM python:3.13-slim

# Install from the source in this repository rather than from PyPI, so the
# image reflects the commit it was built from and a directory introspecting
# it sees the tools that commit actually has.
WORKDIR /app
COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY llm_sidecar ./llm_sidecar
RUN pip install --no-cache-dir .

# Nothing here needs to be root.
RUN useradd --create-home --uid 10001 sidecar
USER sidecar

# stdio transport: the client speaks JSON-RPC on stdin/stdout.
ENTRYPOINT ["llm-sidecar-mcp"]
