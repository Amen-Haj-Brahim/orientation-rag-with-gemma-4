from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
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

    return (
        RunnablePassthrough()
        .assign(source_documents=lambda x: retriever.invoke(x["input"]))
        .assign(context=lambda x: format_docs(x["source_documents"]))
        .assign(
            answer=(
                lambda x: {"context": x["context"], "question": x["input"]}
            )
            | prompt
            | llm
        )
    )


def create_gemma_llm(
    api_key: str,
    model: str = "gemma-4-26b-a4b-it",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        convert_system_message_to_human=True,
    )


def answer_question(rag_pipeline, question: str) -> dict:
    result = rag_pipeline.invoke({"input": question})
    source_documents = []

    if isinstance(result, dict):
        source_documents = result.get("source_documents", [])
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
    }


def format_answer(result: dict) -> str:
    answer = result["answer"]
    source_files = {
        doc.metadata["source_file"]
        for doc in result["source_documents"]
        if "source_file" in doc.metadata
    }

    formatted = f"**Answer:**\n{answer}\n\n"
    if source_files:
        formatted += "**Sources:**\n"
        for source in sorted(source_files):
            formatted += f"- {source}\n"

    return formatted
