## 미세먼지 및 초미세먼지 시계열 예측 Baseline

Dataset Description:
```
\_train
    \_ train_data (npy file)
    \_ train_label (npy file)

# of training samples: 109,136
# of test samples: 5448
```

Feature composition
* 지역: 1개 (값종류: 10 (0 ~ 9))
* 날짜: 3개 (년, 월, 일)
* 이전 5시간동안 미세먼지, 초미세먼지: 5 * 2 = 10개 

Metric: 0.3*미세먼지MSE + 0.7*초미세먼지MSE

Output: 다음시간의 (미세먼지, 초미세먼지)

How to run:

```bash
nsml run -v -d 1_reg_dust -g 1 --memory 12G --shm-size 16G --cpus 4 -e main.py
```

How to list checkpoints saved:

```bash
nsml model ls [NSML ID]/1_reg_dust/[Session Number] (e.g.) nsml model ls KR77777/1_reg_dust/2)
```

How to submit:

```bash
nsml submit -v [NSML ID]/1_reg_dust/[Session Number] [check idx] (e.g) nsml submit -v KR77777/1_reg_dust/2 100)
```
