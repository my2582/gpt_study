import os
import openai
import pinecone
import pickle

from tqdm.auto import tqdm
import datetime
from time import sleep

with open('new_data.pickle', 'rb') as f:
    new_data = pickle.load(f)

EMBEDDING_MODEL = 'text-embedding-ada-002'
openai.api_key = os.getenv('OPENAI_API_KEY')
response = openai.Embedding.create(
    input=[
        'Sample document text goes here',
        'there will be several phrases in each batch'
    ],
    model=EMBEDDING_MODEL
)

print(f'response.keys(): {response.keys()}')
print('len(response[\'data\']: {}'.format(len(response['data'])))
print('len(response[\'data\'][0][\'embedding\']), len(response[\'data\'][1][\'embedding\']): {}, {}'.format(len(response['data'][0]['embedding']), len(response['data'][1]['embedding'])))

index_name = 'openai-youtube-transcriptions'
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment='us-west4-gcp'
)

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(response['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )

index = pinecone.Index(index_name)
index.describe_index_stats()

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(29900, len(new_data), batch_size)):
    # find end of batch
    i_end = min(len(new_data), i+batch_size)
    meta_batch = new_data[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, model=EMBEDDING_MODEL)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=EMBEDDING_MODEL)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'start': x['start'],
        'end': x['end'],
        'title': x['title'],
        'text': x['text'],
        'url': x['url'],
        'published': x['published'],
        'channel_id': x['channel_id']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)
