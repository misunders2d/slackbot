Simple Slack bot which has only one purpose - extract relevant search results from a vector database (Firestore in this case) and summarize them.
Perfect for SOPs, onboarding, storing your specific "tips and tricks" etc.

Requires pre-built vector database with documents having the followign keys: "problem", "solution", "embedding" (the embedding vector)
Uses OpenAI's "embedding" module to encode texts and "chat.completions" to summarize them.
