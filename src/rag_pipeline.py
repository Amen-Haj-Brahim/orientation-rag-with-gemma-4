"""
RAG Pipeline Module

Combines retrieval with Gemma for question answering.
Uses LangChain 1.x API.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough




# Default prompt template for Arabic/English bilingual support
DEFAULT_TEMPLATE = """You are an orientation guidance assistant.
Your task is to answer the question ONLY using the provided Context.

The Context contains CSV rows from an orientation guide.
Answer directly and naturally. Do not mention source files, document numbers, row ids, row indexes, or internal metadata in the answer.
If the answer contains multiple results, present them as a clean list with the useful fields only, such as score, program, university, establishment, duration, requirements, or career paths.
If the information is NOT in the Context, say: "المعلومة غير موجودة في المعطيات."

Context:
{context}

Question: {question}
Answer:"""


def create_rag_pipeline(
    llm: ChatGoogleGenerativeAI,
    retriever,
    template: str = DEFAULT_TEMPLATE
):
    """
    Create a RAG pipeline with retrieval and Gemma.
    
    Args:
        llm: Gemma language model
        retriever: Document retriever
        template: Prompt template
        
    Returns:
        Retrieval chain (RunnableSequence)
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
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
    
    # The chain accepts {"input": question} and returns both answer and sources.
    rag_chain = (
        RunnablePassthrough()
        .assign(
            source_documents=lambda x: retriever.invoke(x["input"])
        )
        .assign(
            context=lambda x: format_docs(x["source_documents"])
        )
        .assign(
            answer=(
                lambda x: {"context": x["context"], "question": x["input"]}
            )
            | prompt
            | llm
        )
    )
    
    return rag_chain


def create_gemma_llm(
    api_key: str,
    model: str = "gemma-4-26b-a4b-it",
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> ChatGoogleGenerativeAI:
    """
    Create Gemini/Google language model.
    
    Args:
        api_key: Google API key
        model: Model name (e.g., gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        convert_system_message_to_human=True
    )


def answer_question(
    rag_pipeline,
    question: str
) -> dict:
    """
    Answer a question using the RAG pipeline.
    
    Args:
        rag_pipeline: Retrieval chain
        question: User question
        
    Returns:
        Dictionary with answer and sources
    """
    result = rag_pipeline.invoke({"input": question})
    source_documents = []

    if isinstance(result, dict):
        source_documents = result.get("source_documents", [])
        result = result.get("answer", "")

    # Extract clean text from response
    if hasattr(result, "content"):
        if isinstance(result.content, list):
            # Find the final text part
            text_parts = [c.get("text", "") for c in result.content if c.get("type") == "text"]
            clean_answer = "\n".join(text_parts)
        else:
            clean_answer = result.content
    else:
        clean_answer = str(result)

    return {
        "answer": clean_answer,
        "source_documents": source_documents
    }


def format_answer(result: dict) -> str:
    """
    Format the answer for display.
    
    Args:
        result: Answer dictionary from answer_question
        
    Returns:
        Formatted answer string
    """
    answer = result["answer"]
    sources = result["source_documents"]
    
    # Get unique sources
    source_files = set()
    for doc in sources:
        if 'source_file' in doc.metadata:
            source_files.add(doc.metadata['source_file'])
    
    formatted = f"**Answer:**\n{answer}\n\n"
    
    if source_files:
        formatted += f"**Sources:**\n"
        for source in sorted(source_files):
            formatted += f"- {source}\n"
    
    return formatted
