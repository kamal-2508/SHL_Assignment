# SHL Assessment Advisor

Conversational agent over the SHL Individual Test Solutions catalog (377 items).
Exposes a stateless FastAPI service: `GET /health`, `POST /chat`.

## Run locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload --port 8000
```

Test:
```bash
curl localhost:8000/health
curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "I am hiring a Java developer"}]
}'
```

## Deploy

Any container host works (Render, Fly, Railway, Modal, Hugging Face Spaces).
Build with the provided `Dockerfile`, set `ANTHROPIC_API_KEY` as an env var on
the host. The service listens on `$PORT` (defaults to 8000).

## Rebuilding the catalog

`data/catalog.json` is pre-built from the provided scrape via
`scripts/build_catalog.py`. Re-run it only if you have a refreshed raw export.

## Structure

```
app/
  main.py       FastAPI app: /health, /chat
  agent.py      Prompting, single LLM call per turn, hallucination guard
  retrieval.py  TF-IDF + alias-boosted search over the catalog
data/catalog.json
scripts/build_catalog.py
```

See `APPROACH.md` for design rationale.
