import time

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


DEFAULT_TEMPLATE = """You are an orientation guidance assistant.
Answer the question only from the context.

Do not mention source files, document numbers, row ids, row indexes, or internal metadata in the answer.
If there are multiple results, use a clean list with useful fields only.
If the information is not in the context, say: "المعلومة غير موجودة في المعطيات."

Context:
{context}

Question: {question}
Answer:"""


def create_rag_pipeline(llm: ChatGoogleGenerativeAI, retriever, template: str = DEFAULT_TEMPLATE):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def format_docs(docs):
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            source = metadata.get("source_file", "unknown")
            row_id = metadata.get("row_id", metadata.get("row_index", "unknown"))
            code = metadata.get("program_code", "unknown")
            formatted_docs.append(
                f"[Document {i} | source={source} | row={row_id} | code={code}]\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(formatted_docs)

    def run(inputs):
        question = inputs["input"]
        timings = {}

        start = time.perf_counter()
        source_documents = retriever.invoke(question)
        timings["retrieval_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        context = format_docs(source_documents)
        prompt_value = prompt.invoke({"context": context, "question": question})
        timings["prompt_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        answer = llm.invoke(prompt_value)
        timings["llm_seconds"] = time.perf_counter() - start
        timings["total_seconds"] = sum(timings.values())

        return {
            "answer": answer,
            "source_documents": source_documents,
            "timings": timings,
        }

    return run


def create_gemma_llm(
    api_key: str,
    model: str = "gemma-4-26b-a4b-it",
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        convert_system_message_to_human=True,
    )


def answer_question(rag_pipeline, question: str) -> dict:
    if hasattr(rag_pipeline, "invoke"):
        result = rag_pipeline.invoke({"input": question})
    else:
        result = rag_pipeline({"input": question})
    source_documents = []
    timings = {}

    if isinstance(result, dict):
        source_documents = result.get("source_documents", [])
        timings = result.get("timings", {})
        result = result.get("answer", "")

    if hasattr(result, "content"):
        if isinstance(result.content, list):
            text_parts = [c.get("text", "") for c in result.content if c.get("type") == "text"]
            clean_answer = "\n".join(text_parts)
        else:
            clean_answer = result.content
    else:
        clean_answer = str(result)

    return {
        "answer": clean_answer,
        "source_documents": source_documents,
        "timings": timings,
    }


def format_answer(result: dict) -> str:
    answer = result["answer"]
    timings = result.get("timings", {})
    source_files = {
        doc.metadata["source_file"]
        for doc in result["source_documents"]
        if "source_file" in doc.metadata
    }

    formatted = f"**Answer:**\n{answer}\n\n"
    if timings:
        formatted += "**Timings:**\n"
        for name, value in timings.items():
            formatted += f"- {name.replace('_seconds', '')}: {value:.2f}s\n"
        formatted += "\n"

    if source_files:
        formatted += "**Sources:**\n"
        for source in sorted(source_files):
            formatted += f"- {source}\n"

    return formatted
