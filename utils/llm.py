from ollama import chat

def generate_answer(question, context):

    prompt = f"""
    You are a financial assistant.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


def summarize_document(text):

    prompt = f"""
    Summarize this financial document.

    Include:
    - Company overview
    - Revenue and profit information
    - Key business segments
    - Major risks
    - Future outlook

    Document:
    {text[:12000]}
    """

    response = chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]