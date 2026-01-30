from openai import OpenAI
from tournament_reranker import make_openai_chat_ranker, rerank_passages

client = OpenAI()  # needs OPENAI_API_KEY in env
ranker = make_openai_chat_ranker(client, model="gpt-4o-mini")

passages = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "Toronto is the capital of Ontario.",
]

ranks = rerank_passages(
    query="Where is the Eiffel Tower?",
    passages=passages,
    ranker=ranker,
    target_k=2,
)

for passage, rank in zip(passages, ranks):
    print(f"rank {rank}: {passage}")
