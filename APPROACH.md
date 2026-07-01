# SHL Assessment Advisor — Approach Document

## What We Built

A conversational API service with two endpoints: GET /health and POST /chat. The service takes a full conversation history and returns the agent's next reply along with a shortlist of SHL Individual Test Solutions. The agent can clarify vague questions, recommend 1 to 10 assessments from the catalog, update the shortlist when the user changes their mind, compare assessments using catalog facts and refuse questions that are outside its scope.

---

## Design Choices

**One API call per turn, no memory on the server.**
Every chat request includes the full conversation history. The server stores nothing between calls. This keeps the design simple and avoids session management bugs. We make one LLM call per turn instead of running a multi-step agent loop. This keeps response times short and well within the 30 second deadline.

**Tech stack: FastAPI + Pydantic + scikit-learn + Groq.**
FastAPI handles the API routing. Pydantic checks that every response matches the required schema before it goes out. We use scikit-learn for catalog search and Groq to run the language model. The model we settled on is llama-4-scout-17b which gives good quality answers in 2 to 5 seconds on the free tier.

---

## How Retrieval Works

The 377 catalog items are cleaned and saved into data/catalog.json. Each item stores the name, URL, description, test type codes, job levels, languages and duration. When a user sends a message we search the catalog using TF-IDF cosine similarity. The item name is weighted twice as heavily as other fields because it is the most important signal. We take the top 20 results and include them in the system prompt as the only source the model is allowed to use for recommendations.

We also added domain keyword boosting. If the conversation mentions words like "sales" or "healthcare" or "graduate" we add extra related terms to the search query. This helps niche assessments rank higher without us having to hardcode specific product names.

---

## How the Prompt Works

The system prompt was written by studying all 10 labeled conversation traces. The main rules are:

- Ask only one clarifying question per turn if the query is too vague
- Recommend immediately if a job description is provided
- When the user says add or remove something keep all other items and only change what was asked
- Answer comparison questions using only facts from the catalog not from general knowledge
- Always include Verify G+ for cognitive ability and OPQ32r for personality on professional roles unless the user says otherwise
- If the user asks to replace OPQ32r with something shorter say there is no shorter personality alternative in the catalog
- Set end of conversation to true only when the user confirms they are done

We also added a turn pressure rule. If the user has answered 3 or more times and there is still no shortlist the agent must recommend now and state any assumptions. This prevents the conversation from running into the 8 turn limit without ever giving a recommendation.

---

## Guardrails Built Into the Code

We do not trust the LLM output blindly. After every response we run these checks in code:

- Every recommended URL is checked against the catalog. If it does not exist it is removed silently
- The test_type code for each item is overwritten using the actual catalog data so it is always correct
- The recommendations list is capped at 10 items
- Any extra JSON keys the model adds are removed
- If the reply text is empty we replace it with a fallback sentence

This means schema compliance does not depend on the model behaving perfectly.

---

## What Did Not Work and How We Fixed It

**Response times were too slow.**
The 70b model took 30 to 40 seconds per call which breaks the 30 second timeout. The 8b instant model was fast but kept hitting the free tier token limit per minute. We tested mixtral which was inconsistent with JSON output. We switched to llama-4-scout-17b which runs in 2 to 5 seconds and has higher token limits.

**Two LLM calls per turn caused problems.**
We tried making a first LLM call to rewrite the search query into richer terms before the main agent call. This doubled the response time and the second call often hit rate limits. We dropped this approach and replaced it with simple domain keyword boosting in the retrieval layer which solves the same problem without any extra LLM calls.

**Verify G+ and OPQ32r were missing from recommendations.**
Early versions did not include these by default even for roles where the labeled traces always include them. We added explicit instructions in the prompt and a code level fallback so the reply is never empty.

**Wrong test type codes.**
The model sometimes guessed wrong codes like C for Verify G+ instead of A. We fixed this by overwriting the test_type field after every response using the catalog's ground truth data.

**Some assessments never appeared in results.**
Assessments like SVAR, GSA, Medical Terminology and DSI were not ranking in the top 20 for some queries. We fixed this with domain keyword boosting which adds relevant terms to the search when certain topics are detected in the conversation.

---

## How We Evaluated

We replayed all 10 public conversation traces through the /chat endpoint and checked that the shortlist at each turn matched the expected items. We measured response time using curl to confirm all calls stayed under 30 seconds. We also ran automated unit tests with a mocked LLM to verify schema compliance, URL validation, turn pressure and refusal behavior.

---

## AI Tools Used

Claude (claude.ai) was used as a coding assistant to scaffold the FastAPI service, retrieval module and initial prompt drafts.

All design decisions were made by me based on reading the spec and testing against the 10 traces. I directed Claude to implement specific fixes after I identified problems such as switching models when responses were too slow, dropping the two-call approach when it caused rate limit errors and adding domain keyword boosting when niche assessments were not appearing. The debugging, testing, timing and evaluation were all done by me. Claude was a tool not a replacement for judgment.
