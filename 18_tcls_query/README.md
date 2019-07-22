# NIPA Query Similarity Baseline
NIPA AI STARTHON 2019 Query Similarity Baseline

## Data
- Name: ```18_tcls_query```
- Label: binary
  - 1 for similar query pairs
  - 0 for nonsimilar query pairs
- Train data: 115,198 query pairs
  - train_data/train_data
  - train_data/train_label
- Validation data: 20,000 query pairs
  - train_data/valid_data
  - train_data/valid_label
- Test data: 20,000 query pairs

### Example

#### train_data
Tab separated file, with format `[SOURCE]\t[QUERY_A]\t[QUERY_B]`. Some examples have unknown source, and have `_` as the source.
```
2068803_388327	광주에서 부산까지 고속도로 톨게이트 요금 	14일 전국 고속도로 요금
_	급해요 !!진짜로 !!	급해요 진짜로요..
```
#### train_label
Binary similarity score for each example in the data file.
```
0
1
```

## Requirements

`numpy` must be installed on your local environment.

## Usage
### Train

```bash
nsml run -d 18_tcls_query -e main.py
```

### List saved checkpoints

```bash
nsml model ls SESSION
```

### Submit

```bash
nsml submit SESSION CHECKPOINT
```

## Metric
ROC-AUC Score
