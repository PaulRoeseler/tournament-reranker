from openai import OpenAI
from tournament_reranker import make_openai_chat_ranker, rerank_passages

client = OpenAI()  # needs OPENAI_API_KEY in env
ranker = make_openai_chat_ranker(client, model="gpt-4o-mini")

passages = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "Toronto is the capital of Ontario.",
]

top_chunks = rerank_passages(
    query="Where is the Eiffel Tower?",
    passages=passages,
    ranker=ranker,
    target_k=2,
)

for c in top_chunks:
    print(c.text, c.metadata, c.base_rank)
