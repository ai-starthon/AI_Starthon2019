# 생산 공정에서 진동 신호에 의한 공정 불량 진단

### 1. Usage

#### How to run

```
nsml run -d 3_cls_crane2
```

#### How to check session logs
```
nsml logs -f [SESSION_NAME] # e.g., nsml logs -f nsmlteam/3_cls_crane2/1
```

#### How to list checkpoints saved
You can search model checkpoints by using the following command:
```
nsml model ls nsmlteam/3_cls_crane2/1
```

#### How to submit
The following command is an example of running the evaluation code using the model checkpoint at 10th epoch.
```
nsml submit  nsmlteam/3_cls_crane2/1 1
```

#### How to check leaderboard
```
nsml dataset board 3_cls_crane2
```
