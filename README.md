# SHL Assessment Advisor

Conversational agent over the SHL Individual Test Solutions catalog (377 items).
Exposes a stateless FastAPI service: GET /health, POST /chat.

## Run locally

1. Install dependencies:
pip install -r requirements.txt

2. Add your Groq API key to .env:
GROQ_API_KEY=gsk_your_key_here

3. Start the server:
uvicorn app.main:app --reload --port 8000

4. Test:
curl localhost:8000/health
curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "I am hiring a Java developer"}]}'

## Deploy to Render

1. Push to GitHub
2. Create a new Web Service on Render, connect your repo
3. Set GROQ_API_KEY as an environment variable
4. Render auto-detects the Dockerfile and deploys

## Rebuilding the catalog

data/catalog.json is pre-built from the raw SHL scrape.
To rebuild: update SRC in scripts/build_catalog.py to point to your raw JSON, then run:
python3 scripts/build_catalog.py

## Project structure

app/main.py — FastAPI app, /health and /chat endpoints
app/agent.py — System prompt, LLM call, hallucination guard
app/retrieval.py — TF-IDF search over the catalog
data/catalog.json — 377 cleaned SHL Individual Test Solutions
scripts/build_catalog.py — Cleans raw scrape into catalog.json
Dockerfile, requirements.txt, APPROACH.md

## Model

Uses meta-llama/llama-4-scout-17b-16e-instruct via Groq free tier.
Typical response time: 2-5 seconds. Well within the 30 second timeout.

See APPROACH.md for full design rationale.
