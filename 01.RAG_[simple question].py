import os
import openai
from typing import Union, List

openai.api_key = os.getenv('OPENAI_API_KEY')
COMPLETION_MODEL = 'text-davinci-003'

query = 'who was the 12th person on the moon and when did they land?'


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


answer = complete(prompt=query)
print(f'Query: {query}')
print(f'Answer: {answer}')
