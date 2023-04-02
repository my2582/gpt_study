from datasets import load_dataset
from tqdm.auto import tqdm
import pickle

# ML, tech 관련 유튜브 채널에서 음성을 텍스트로 변환시킨 데이터 로드
data = load_dataset('jamescalam/youtube-transcriptions', split='train')

new_data = []
window = 20     # 한 Chuck로 병합될 문장 개수
stride = 4      # Chucks간 중복되는 문장 개서 ("stride over")

for i in tqdm(range(0, len(data), stride)):
    i_end = min(len(data)-1, i+window)
    if data[i]['title'] != data[i_end]['title']:
        continue  # 다음 비디오가 나오면 new_data에 append 중단하고 다음으로 넘어감
    text = ' '.join(data[i:i_end]['text'])
    new_data.append({
        'start': data[i]['start'],
        'end': data[i_end]['end'],
        'title': data[i]['title'],
        'text': text,
        'id': data[i]['id'],
        'url': data[i]['url'],
        'published': data[i]['published'],
        'channel_id': data[i]['channel_id']
    })

    with open('new_data.pickle', 'wb') as f:
        pickle.dump(new_data, f)

