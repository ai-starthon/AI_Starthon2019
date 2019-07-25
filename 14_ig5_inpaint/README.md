# 14_ig5_inpaint

## Task description
For given input images and their corresponding masks, synthesize GT images.

## Evaluation metric
L1 distance in RGB space between synthesized image and GT

## example data
input, mask, GT

<img width=200 src="https://github.com/ai-starthon/AI_Starthon2019/blob/master/14_ig5_inpaint/example/x_input.png"/> <img width=200 src="https://github.com/ai-starthon/AI_Starthon2019/blob/master/14_ig5_inpaint/example/x_mask.png"/> <img width=200 src="https://github.com/ai-starthon/AI_Starthon2019/blob/master/14_ig5_inpaint/example/x_GT.png"/>

## stats

| Name of the dataset | 14_ig5_inpaint |
| - | - |
| Number of train data | 48000 | 
| Number of test data | 10000 | 


## How to run:

```bash
nsml run -d 14_ig5_inpaint -e main.py 
```

## How to list checkpoints saved:

```bash
# nsml model ls YOUR_ID/14_ig5_inpaint/SESSION_NUM
nsml model ls nipachallenger/14_ig5_inpaint/1

```

## How to submit:

```bash
# nsml submit  YOUR_ID/14_ig5_inpaint/SESSION_NUM CKPT_EPOCH
nsml submit  nipachallenger/14_ig5_inpaint/1 9
```
