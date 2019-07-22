# NIPA_FACE_FEWSHOT_CLS
NIPA few shot face classification challenge

Dataset: `7_icls_face`
* num_train_images: 180K
* num_train_classes: 100 (60: full images - about 3000 images per class. 40: few images - about 80 images)
* num_test_images: 117K (40: full images)
* num_test_classes: 40 (Note that few samples of each class are included in the training datset)

Datset name format:
* 17080803_S001_L01_E01_C1.jpg describes
* name_accessary_Light_Emotion_Angle

Options
'''
num Accessary: 6
num Light: 10 
num Emotion: 3
num Angle: 20
'''

For the ordinary 60 classes in training datset, they have the images with all the different options.
We note that for the 40 number of few shot classes, we select only one accessary case (S001, no accessary).
However, in the testset, the 40 classes have the images with all the different options.
Our goal is to classfy all the images in the testset, given the information of remaining 60 classes.

The face is cropped by a face detector, and the number of images for each classes can be slightly different.

The baseline code implemented is a simple classification code, and of course, does not work well in the test scenario.


Pushing dataset (at the root of the folder you want to upload):

```bash
nsml dataset push -v -f -l 7_icls_face /local/path/to/dataset/
```

How to run:

```bash
nsml run -v -d 7_icls_face -g 1 --memory 12G --shm-size 32G --cpus 10 -e main.py
```

How to list checkpoints saved:

```bash
nsml model ls KR18588/7_icls_face/1
```

How to submit:

```bash
nsml submit -v KR18588/7_icls_face/1 10
```
