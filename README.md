# [밑바닥부터 만들면서 배우는 LLM](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=372272431)
> - 지은이 : [Sebastian Raschka](https://github.com/rasbt)
> - 옮긴이 : [박해선](https://github.com/rickiepark)
> - 출판사 : [길벗](https://github.com/gilbutITbook)

[<밑바닥부터 만들면서 배우는 LLM> 완독챌린지](https://www.inflearn.com/challenge/lt밑바닥부터-만들면서-배우는-llm)에 참여하여 7주간 책읅 읽고 예제코드를 따라하면서 느낀점 작성

## 환경 설정
- conda create -n llm-from-scratch python=3.11
- conda activate llm-from-scratch
- pip install -r requirements.txt

## [1장 : 대규모 언어 모델 이해하기](https://github.com/hwangintae/llm-from-scratch/pull/1)
GPT가 Transformer를 이용한다는 것은 알고 있었는데 decoder만 사용하는지 몰랐다.

few-shot, zero-shot에 대해 듣긴했지만 정확히 무엇을 뜻하는지 몰랐다.

BERT와 GPT를 사용하고 있지만 둘의 차이에 대해 알게된 것은 처음이다.

비교적 짧은 시간을 투자해서 다양한 내용을 알게되어 기쁘고 앞으로의 chapter가 기대된다.

## [2장 : 텍스트 데이터 다루기](https://github.com/hwangintae/llm-from-scratch/pull/2)
임베딩에 대해서 다루었다. 책 내용 중에 '셀프 어텐션 메커니즘 자체가 위치에 구애받지 않기...'가

무슨 말인지 사실 이해가 잘 되지 않는다. 아직 잘 모르기 때문인거 같은데 글자 그대로 이해하기론

위치를 고려한 임베딩 방법이 2가지가 있고, 이것은 처리하는 데이터에 성질에 따라 달라지는 경우가 있다고 하는데

잘 모르겠다.

강의를 들으면서 따라가야겠다.


## [3장 : 어텐션 메커니즘 구현하기](https://github.com/hwangintae/llm-from-scratch/pull/3)
딥러닝에 대한 기본적인 내용은 알고 있어서 sotfmax, dropout에 대한 사전지식은 있었지만,

self attention 자체는 처음이라 이해하는데 어려웠다.

RNN은 이전 은닉층의 데이터를 알 수 없어서 맥락을 이해할 수 없지만,
self attention는 입력에 대한 attention score, attention weight를 이용해 문맥을 이해할 수 있다.

정도만 이해했다.

attention에 대해 이해를 하기 위해선 책을 여러번 읽어야겠다.
## 4장 : 밑바닥부터 GPT 모델 구현하기
## 5장 : 레이블이 없는 데이터를 활용한 사전 훈련
## 6장 : 분류를 위해 미세 튜닝하기
## 7장 : 지시를 따르도록 미세 튜닝하기
