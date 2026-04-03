from __future__ import annotations

from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful YouTube video assistant.\n"
        "Answer the question using only the transcript context below.\n"
        "If the answer is not in the transcript, say that clearly.\n\n"
        "Transcript context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)


def load_text_generator(model_id: str, hf_token: str):
    return InferenceClient(model=model_id, token=hf_token)


def answer_question(
    generator,
    vector_store,
    question: str,
    top_k: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, list]:
    docs = vector_store.similarity_search(question, k=top_k)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    result = generator.text_generation(
        prompt,
        model=generator.model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_full_text=False,
    )
    answer = result.strip()
    return answer, docs
