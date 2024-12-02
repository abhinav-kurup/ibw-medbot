

prompt_template="""
Use the following pieces of symptoms to get the disease or cause of these symptoms.
give multiple diseases(maximum three), separate them with a comma from the most possible to the least possible.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""