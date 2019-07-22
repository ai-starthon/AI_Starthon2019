# NIPA_Korean_Food_CLS
NIPA kroean food classification challenge

Dataset: `4_cls_food`
* num_train_images: 120K
* num_train_classes: 150
* num_test_images: 1197
* num_test_classes: 113
* All test classes are subsets of train classes.

Pushing dataset (at the root of the folder you want to upload):

```bash
nsml dataset push -v -f -l -e evaluation.py 4_cls_food /local/path/to/dataset/
```

How to run:

```bash
nsml run -v -d 4_cls_food -g 1 --memory 12G --shm-size 32G --cpus 10 -e main.py
```

How to list checkpoints saved:

```bash
nsml model ls KR18588/4_cls_food/1
```

How to submit:

```bash
nsml submit -v KR18588/4_cls_food/1 10
```
