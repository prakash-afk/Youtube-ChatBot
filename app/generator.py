from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


PROMPT_TEMPLATE = (
    "You are a helpful YouTube video assistant.\n"
    "Answer the question using only the transcript context below.\n"
    "If the answer is not in the transcript, say that clearly.\n\n"
    "Transcript context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)


def load_text_generator(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


def answer_question(generator, vector_store, question: str, top_k: int, max_new_tokens: int, temperature: float) -> tuple[str, list]:
    docs = vector_store.similarity_search(question, k=top_k)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
    )
    answer = result[0]["generated_text"].strip()
    return answer, docs
