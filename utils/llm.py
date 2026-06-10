from ollama import chat


def generate_answer(question, context):

    prompt = f"""
    You are an expert financial analyst.

    Use ONLY the provided context to answer.

    When answering:

    - Explain financial metrics if relevant.
    - Mention trends when available.
    - Highlight risks when relevant.
    - Be concise and professional.
    - Do not make up information.

    If the answer is not present in the context,
    say:
    "I could not find that information in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def summarize_document(text):

    prompt = f"""
    Summarize this financial document.

    Include:

    - Company overview
    - Revenue highlights
    - Profitability highlights
    - Key business segments
    - Major risks
    - Future outlook

    Document:
    {text[:12000]}
    """

    response = chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def financial_analysis(text):

    prompt = f"""
    You are a senior financial analyst.

    Analyze the following financial document.

    Provide:

    1. Revenue Analysis
    2. Profitability Analysis
    3. Risk Assessment
    4. Growth Opportunities
    5. Overall Financial Health

    Give actionable insights where possible.

    Document:
    {text[:12000]}
    """

    response = chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]