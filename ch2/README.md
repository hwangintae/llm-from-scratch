# 텍스트 데이터 다루기
- Transformer는 ChatGPT에서 사용되는 모델 및 GPT와 요사한 LLM의 기반이 된다.
- LLM 모델을 추가적으로 미세 튜닝하여 일반적인 지시를 따르거나 특정 작업을 수행하도록 만들 수 있다.
- LLM을 구현하고 훈련하기 전에 훈련 데이터셋을 준비해야 한다.

## 2.1 단어 임베딩 이해하기
- 텍스트는 범주형 데이터 이므로 신경망을 훈련하는 데이터로 바로 사용하지 못한다.
- 단어를 실수 vector로 표현할 방법이 필요하다.
- 데이터 포맷마다 고유한 임베팅 모델이 필요하다.
  - 텍스트를 위한 임베팅
  - 오디오를 위한 임베딩 등..
- **임베딩의 주요 목적은 신경망이 처리할 수 있는 포맷으로 변환하는 것**
- **Word2Vec**는 문맥 단어를 예측하거나 그 반대의 방식으로 신경망을 훈련하여 단어 임베딩을 생성
  - **핵심 아이디어는 비슷한 맥락에 등장하는 단어는 비슷한 의미를 가지는 경향이 있다.**
- LLM은 Word2Vec을 사용하는 대신 훈련의 일부로 임베딩을 최적화하면 임베딩을 특정 작업과 주어진 데이터에 최적화할 수 있다.

## 2.2 텍스트 토큰화하기
- LLM 훈련을 위해 토큰화할 텍스트는 공개된 단편 소설로 하겠다.
```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("총 문자 개수:", len(raw_text))
print(raw_text[:99])
```
> [!Note]
> 교육적인 목적을 위해 작은 텍스트 샘플로 토큰화를 한다.

- re를 사용해서 텍스트를 토큰 리스트로 분할 (나중에 사전 훈련된 토크나이저를 사용)
```python
import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
```

## 2.3 토큰을 토큰 ID로 변환하기
- 문자열에서 정수 표현으로 바꿔 토큰 ID를 만드는 과정은 임베딩 벡터로 변환하기 전의 중간 단계
- 토큰을 토큰 ID로 매핑하려면 어휘사전을 구축해야 한다.
```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
```
- 간단한 토큰나이저 만들기
```python
import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text
```
```python
from tokenizer import SimpleTokenizerV1

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)

print(ids)
print(tokenizer.decode(ids))
```

## 특수 문맥 토큰 추가하기
- 알지 못하는 단어를 처리하기 위해 토크나이저 수정
- <|unk|>, <|endoftext|> 2개의 토큰을 지원하도록 어휘사전과 토크나이저 수정
- 어휘사전 추가
```python
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))
```
- 토크나이저 추가
```python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text
```
- LLM에 따라 다음과 같은 특수 토큰을 사용하기도 한다.
  - [BOS] beginning of sequence : 텍스트의 시작을 표시
  - [EOS] end of sequence : 텍스트의 끝
  - [PAD] padding : 모든 텍스트의 길이를 동일하게 맞추기 위해

## 바이트 페어 인코딩
- 바이트 페어 인코딩(이하 BPE)는 상대적으로 복잡하기 때문에 tiktoken(0.9.0) 라이브러리를 사용
```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)
```
- BPE 특징
  - <|endoftext|> 토큰은 50256 같이 비교적 큰 토큰 ID에 할당
  - someunknownPlace와 같은 알지 못하는 단어를 정확하게 인코딩하고 디코딩
  - 어휘사전에 없는 단어를 더 작은 부분단어로(심지어 개별 문자로) 나누어 처음 본 단어를 처리
  - LLM이 훈련 데이터에 없는 단어가 포함되어 있더라도 텍스트를 처리할 수 있다.
  - 반복적으로 자주 등장하는 문자를 부분단어로 합치고
  - 다시 자주 등장하는 부분단어를 단어로 합쳐서 어휘사전을 구축한다.
- 연습문제 2.1
```python
exercise_text = "Akwire ier"

exercise_encode = tokenizer.encode(exercise_text)
print(exercise_encode)
# [33901, 21809, 220, 959]

exercise_decode = tokenizer.decode(exercise_encode)
print(exercise_decode)
# Akwire ier

# encoding 결과가 다르지만 decode의 결과는 같다.
```

## 슬라이딩 윈도로 데이터 샘플링하기
- LLM은 텍스트에 있는 다음 단어를 예측하는 식으로 사전 훈련된다.
- BPE으로 The Verdict 전체를 토큰화
```python
import tiktoken

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
tokenizer = tiktoken.get_encoding("gpt2")

enc_text = tokenizer.encode(raw_text)
```
- 다음 단어 예측 작업을 위해 입력 토큰을 담은 x와 입력에서 토큰 하나만큼 이동한 타깃을 담은 y를 만듬
```python
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")
```
```python
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    
    print(context, "---->", desired)

# [290] ----> 4920
# [290, 4920] ----> 2241
# [290, 4920, 2241] ----> 287
# [290, 4920, 2241, 287] ----> 257
```
```python
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# and ---->  established
# and established ---->  himself
# and established himself ---->  in
# and established himself in ---->  a
```
- 입력 데이터셋을 순회하면서 pytorch tensor로 반환하는 로더를 구현해야한다.
```python
# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def create_dataloader_v1(txt, batch_size=4, max_length=256,stride=128, shuffle=True, drop_last=True, num_workers=0):
        tokenize = tiktoken.get_encoding("gpt2")

        dataset = GPTDatasetV1(txt, tokenize, max_length, stride)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
```
- GPTDatasetV1 class와 create_dataloader_v1 함수가 얻허게 동작하는지 이해하기 위해 문맥 크기 4와 배치 크기 1로 dataloader를 테스트
```python
from dataset import GPTDatasetV1

dataloader = GPTDatasetV1.create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
```
- 첫 번째 tensor는 입력 token ID, 두 번째 tensor는 타겟 token ID를 저장하고 있다.
- stride는 스라이딩 윈도가 배치에 걸쳐 입력 위를 이동하는 크기를 지정한다.
- 연습문제
```python
# 연습문제
dataloader = GPTDatasetV1.create_dataloader_v1(raw_text, batch_size=1, max_length=2, stride=2, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# [tensor([[ 40, 367]]), tensor([[ 367, 2885]])]

dataloader = GPTDatasetV1.create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
first_batch = next(data_iter)
print(first_batch)

# [tensor([[  40,  367, 2885, 1464, 1807, 3619,  402,  271]]), tensor([[  367,  2885,  1464,  1807,  3619,   402,   271, 10899]])]
# [tensor([[ 2885,  1464,  1807,  3619,   402,   271, 10899,  2138]]), tensor([[ 1464,  1807,  3619,   402,   271, 10899,  2138,   257]])]

# 연습문제
dataloader = GPTDatasetV1.create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
first_batch = next(data_iter)
print(first_batch)

# [tensor([[  40,  367, 2885, 1464, 1807, 3619,  402,  271]]), tensor([[  367,  2885,  1464,  1807,  3619,   402,   271, 10899]])]
# [tensor([[  367,  2885,  1464,  1807,  3619,   402,   271, 10899]]), tensor([[ 2885,  1464,  1807,  3619,   402,   271, 10899,  2138]])]
```

## 2.7 token 임베딩 만들기
- 간단한 설명을 위해 6개 단어로 구성된 어휘사전, 크기가 3인 임베딩을 만듬
```python
import torch

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
```

## 2.8 단어 위치 encoding 하기
- 원칙적으로 token 임베딩은 LLM의 입력으로 적합하다.
- 하지만 LLM의 self Attention 메커니즘 자체가 위치에 구애받지 않기 때문에
- LLM에 추가적인 위치 정보를 주입하는 것이 도움이 된다.
  - 상대 위치 임베딩
    - 토큰 사이의 거리를 강조
    - 이런 방식은 모델이 길이가 다른 시퀀스에도 더 잘 일반화될 수 있다.
  - 절대 위치 임베딩
    - 입력 시퀀스의 각 위치에 대해 고유한 임베딩이 토큰 임베딩에 더해져 정확한 위치 정보를 추가
  - 두 종류의 위치 임베딩은 LLM이 토큰 사이의 순서와 관계를 이해하는 능력을 보강하여
  - 정확하고 맥락을 고려한 예측을 만드는 데 목적이 있다.
- 입력 토큰을 256 차원의 벡터 표현으로 인코딩해 보고, 50,257 크기의 어휘사전을 가진 BPE 토크나이저로 만들었다고 가정
```python
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```
```python
from dataset import GPTDatasetV1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
dataloader = GPTDatasetV1.create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("토큰 ID:\n", inputs)
print("\n입력 크기:\n", inputs.shape)

# 토큰 ID:
# tensor([[   40,   367,  2885,  1464],
#         [ 1807,  3619,   402,   271],
#         [10899,  2138,   257,  7026],
#         [15632,   438,  2016,   257],
#         [  922,  5891,  1576,   438],
#         [  568,   340,   373,   645],
#         [ 1049,  5975,   284,   502],
#         [  284,  3285,   326,    11]])
# 
# 입력 크기:
# torch.Size([8, 4])
```
- GPT 모델의 절대 임베딩 방법에서는 token_embedding_layer와 동일한 임베딩 차원을 가지는 또 다른 임베딩 층을 만든다.
```python
context_length = max_length

pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

print(pos_embeddings.shape)
```
- context_length는 LLM이 지원하는 입력 크기를 나타내는 변수