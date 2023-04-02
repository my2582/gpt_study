import os
import openai
from typing import Union, List
import pinecone

query = (
    'Which training method should I use for sentence transformers when ' +
    'I only have paris of related sentences?'
)

# init: Pinecone index
index_name = 'openai-youtube-transcriptions'

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment='us-west4-gcp'
)

# create: OpenAI embedding
openai.api_key = os.getenv('OPENAI_API_KEY')
COMPLETION_MODEL = 'text-davinci-003'
EMBEDDING_MODEL = 'text-embedding-ada-002'
OPENAI_LIMIT = 3750



# Connect to my Pinecone index.
index = pinecone.Index(index_name)

# view index stats
index.describe_index_stats()


def retrieve(query: Union[str, List]):
    response = openai.Embedding.create(
        input=[query],
        model=EMBEDDING_MODEL
    )

    xq = response['data'][0]['embedding']

    # Get relevant documents through Pinecone retrieval (including the questions)
    response = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in response['matches']
    ]

    # Build our prompt with the retrieved contexts included:
    prompt_start = (
        'Answer the question based on the context below.\n\n' +
        'Context:\n'
    )
    prompt_end = (
        f'\n\nQuestion: {query}\nAnswer: '
    )

    # Append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len('\n\n---\n\n'.join(contexts[:i])) >= OPENAI_LIMIT:
            prompt = [
                prompt_start +
                '\n\n---\n\n'.join(contexts[:i-1]) +
                prompt_end
            ]
            break
        elif i == len(contexts) - 1:
            prompt = [
                prompt_start +
                '\n\n---\n\n'.join(contexts) +
                prompt_end
            ]

    return prompt


def complete(prompt: Union[str, List]) -> str:
    response = openai.Completion.create(
        model=COMPLETION_MODEL,
        prompt=prompt,
        temperature=0,          # RF에서 explore 확률값로 해석 중
        max_tokens=400,
        top_p=1,                # 토큰풀의 크기를 나타낸다고 해석 중(big pool if p=1)
        frequency_penalty=0,    # 값이 클수록 아웃풋에 출현 빈도수가 높은 토큰에 패널티를 부여함으로써 동의어나 유사어 사용을 유도함.
        presence_penalty=0,     # 값이 클수록 아웃풋에 이미 나타난 토큰을 재사용할 경우(whether they appear or not) 패널티를 주는 방식으로
                                # 새로운 토픽 생성을 유도함
        stop=None
    )
    return response['choices'][0]['text'].strip()


# First, we retrieve relevant items for Pinecone.
query_with_contexts = retrieve(query=query)
print(query_with_contexts)

answer = complete(query_with_contexts)
print(f'Query: {query}')
print(f'Answer: {answer}')
