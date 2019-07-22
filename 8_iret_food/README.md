# 음식 사진 조회

## 문제설명

음식 사진이 주어졌을 때 (쿼리 사진), 음식 사진 데이터베이스에서 같은 음식 카테고리를 담고 있는 사진 1장을 조회하는 문제입니다.

|| 쿼리 사진 | 데이터베이스 사진 (맞는 답) | 데이터베이스 사진 (틀린 답)  |
| :---: | :---: | :---: | :---: |
| 이미지 | <img src="https://oss.navercorp.com/nsml/nipa/raw/master/8_iret_foot/example_images/Img_082_0042.jpg" width="200">  | <img src="https://oss.navercorp.com/nsml/nipa/raw/master/8_iret_food/example_images/Img_082_0047.jpg" width="200"> | <img src="https://oss.navercorp.com/nsml/nipa/raw/master/8_iret_food/example_images/Img_069_0070.JPG" width="200"> |
| ImageID | Img_082_0042.jpg | Img_082_0047.jpg | Img_069_0070.JPG |

## 데이터
Dataset: `8_iret_food`
* num_train_images: 120K
* num_train_classes: 150
* num_test_images: 1197
* num_test_classes: 113
* All test classes are subsets of train classes.

## 평가 지표

평가 데이터셋을 다음과 같이 표현합니다 - `x1,...,xN`. 여기서 첫번째 이미지 `x1`을 쿼리 사진으로, 나머지 `x2,...,xN`를 데이터베이스 사진으로 생각하고 `x2,...,xN` 중에서 `x1`과 같은 카테고리의 음식을 담고 있을 것 같은 사진을 선택합니다 (CNN feature 들의 내적 연산등을 통해 선택). 만약 실제 카테고리가 맞다면 쿼리 `x1`은 정답으로 처리합니다.

같은 방식으로 `x2`를 쿼리로, 나머지 `x1,x3,...,xN`을 데이터베이스로 하여 `x2`에 대한 음식 조회의 정답 여부를 판별합니다. 이 과정을 (leave-one-out query selection) 을 `N` 회 전체 평가셋 `x1,...,xN`에 대하여 진행하면 쿼리별로 정답 여부가 결정됩니다. 이때 정확도 (top-1 accuracy) 를 다음과 같이 계산합니다.

```
accuracy = (평가셋 중 맞았던 쿼리 개수) / (평가셋에 있는 이미지 개수) 
```

이 수치가 리더보드에 표시됩니다.

## 베이스라인 모델

주어진 베이스라인 방법은 (1) `ResNet18` 을 학습 셋에 있는 카테고리 레이블을 이용해 Cross-entropy loss 로 
학습시킨 후 (2) 각 인풋 이미지에 대해 마지막에서 두번째 CNN 레이어 feature를 계산하고
(3) L2 normalization 정규화를 시켜서 벡터 내적으로 쿼리 이미지와 데이터베이스 이미지들
사이의 유사도를 계산, (4) 데이터베이스에서 가장 가까운 이미지의 ID 를 유추합니다.

`main.py` 파일을 참고하여 위 파이프라인을 변형/새로구현 하면 됩니다. 기본적으로
`_infer` 함수와 `ImplementYourself` 클라스 아래의 메쏘드와 
`if __name__ == "__main__":` 내의 트레이닝 코드를 변형하게 될 것입니다.

베이스라인 모델 학습 시작 방법

```bash
nsml run -v -d 8_iret_food -g 1 --memory 12G --shm-size 32G --cpus 10 -e main.py
```

학습이 시작되면 `{USER_ID}/8_iret_food/{SESSION_NUMBER}` 정보가 출력됩니다.
이것을 이용해 모델 체크포인트의 리스트를 확인할 수 있으며

```bash
nsml model ls {USER_ID}/8_iret_food/{SESSION_NUMBER}
```

모델을 평가 서버에 보내서 리더보드에 올릴 수 있습니다.

```bash
nsml submit -v {USER_ID}/8_iret_food/{SESSION_NUMBER} {CHECKPOINT_NUMBER}
```

