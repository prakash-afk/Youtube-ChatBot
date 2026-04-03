from __future__ import annotations

import torch
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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


def load_text_generator(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if torch.cuda.is_available() else None,
    )


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


def truncate_context(context: str, max_context_chars: int) -> str:
    if len(context) <= max_context_chars:
        return context
    return context[:max_context_chars].rsplit(" ", 1)[0]


def answer_question(
    generator,
    retriever,
    question: str,
    max_context_chars: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, list]:
    docs = retriever.invoke(question)
    context = truncate_context(format_docs(docs), max_context_chars)
    prompt = build_prompt(context, question)

    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    answer = result[0]["generated_text"].strip()
    return answer, docs
