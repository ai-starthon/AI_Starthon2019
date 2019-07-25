# nipa-mrc-baseline
NIPA AI STARTHON 2019 MRC Baseline with CLaF

## Overview

- NSML Dataset Name: `19_tcls_qa`
  - Train set - 226894개
  - Dev set - 18116개
  - Test set - 12680개
- Challenge: 기계독해
	- * 기계독해(Machine Reading Comprehension, MRC) : 제시된 본문 안에서 질의에 대한 정답을 추론하는 딥러닝 기반 기술
- Example:

```
{
    "data": [{
        "source": 6,
        "paragraphs": [{
            "qas": [{
                "question": "쎈 마이웨이 관련 기자간담회 누가 했어",
                "id": "m4_278529-1",
                "answers": [{
                    "answer_start": 0,
                    "text": "박영선"
                }]
	}],
        "context": "박영선 PD는 18일 오후 서울 양천구 목동 SBS에서 모비딕의 토크 콘텐츠 쎈 마이웨이 관련 기자간담회를 열고 출연진에 신뢰를 드러냈다."
	}],
	"title": “1 "}
```

- 본문 카테고리(source)의 기입형태 및 설명
    - 1 : 정치
    - 2 : 경제
    - 3 : 사회
    - 4 : 생활
    - 5 : IT/과학
    - 6 : 연예
    - 7 : 스포츠
    - 8 : 문화
    - 9 : 미용/건강

- 육하원칙(classtype)의 기입 형태 및 설명
    - work_where : 어디서
    - work_who : 누가
    - work_what : 무엇을
    - work_how : 어떻게
    - work_why : 왜
    - work_when : 언제


## Usage

- Requirements
    - [claf](https://github.com/naver/claf)>=0.1.6
- Docker Images: [claf/claf:latest](https://hub.docker.com/r/claf/claf/tags)



- Train and Evaluate

```bash
# local
python main.py --base_config base_config/baseline.json

# nsml
nsml run -d 19_tcls_qa -g 1 --memory "50GB" -e main.py -a "--base_config base_config/baseline.json"
```
- Submit
    - Submit 시에는 Vocab을 로딩하는 과정이 필요합니다. 그래서 `def test()` 함수 안에 있는 `NSML_CHECKPOINT`, `NSML_SESSION` 이 두 변수에 제출에 사용하는 세션과 체크포인트를 입력한 후 아래 커맨드를 입력합니다.

```bash
nsml submit -e main.py SESSION_NAME CHECKPOINT 
```
