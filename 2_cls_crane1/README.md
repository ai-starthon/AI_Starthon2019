# 도금설비 진동 데이터 불량 진단

### 1. Usage

#### How to run

```
nsml run -d 2_cls_crane1
```

#### How to check session logs
```
nsml logs -f [SESSION_NAME] # e.g., nsml logs -f nsmlteam/2_cls_crane1/1
```

#### How to list checkpoints saved
You can search model checkpoints by using the following command:
```
nsml model ls nsmlteam/2_cls_crane1/1
```

#### How to submit
The following command is an example of running the evaluation code using the model checkpoint at 10th epoch.
```
nsml submit  nsmlteam/2_cls_crane1/1 1
```

#### How to check leaderboard
```
nsml dataset board 2_cls_crane1
```
