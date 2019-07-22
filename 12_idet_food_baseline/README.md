## 사진 내의 음식 영역 추출하기 Baseline

Task
```
사진 내의 음식 영역 추출하기 
```

Dataset Description:
```
\_train
    \_ train_data (folder)
        \_ *.jpg (음식 사진)
        \_ *.txt (bounding box ([x,y,w,h]))

# of training images: 40,612
# of test images: 1,791
```

Evaluation Metric:
```
mean of intersection-over-union (mIoU)
(1/N)*sum_{k=1}^{N}(IoU(pred_k, gt_k))

- N: number of test images
- pred_k: bouding box prediction ([x,y,w,h]) of k-th image
- gt_k: bouding box ground truth ([x,y,w,h]) of k-th image
```


How to run:

```bash
nsml run -v -d 12_idet_food -g 1 --cpus 2 -e main.py
```

How to list checkpoints saved:

```bash
nsml model ls {USER_ID}/12_idet_food/{SESSION_NUMBER}
```

How to submit:

```bash
nsml submit -v {USER_ID}/12_idet_food/{SESSION_NUMBER} {CHEKCOPOINT_NUMBER}
```

