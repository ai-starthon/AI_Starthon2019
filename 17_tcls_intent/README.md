# NIPA Intent Classification Baseline
NIPA AI STARTHON 2019 Intent Classification Baseline

## Overview
**Data**
- Name: ```17_tcls_intent```
- Number of train data: 40,373
- Number of validation data: 4,057
- Number of test data: 4,057
- Number of intention class: 2,253
- Example:
```
{"intent": "메뉴문의", "utterance": "성분이 홍차인지, 아니면 뭐예요?", "intent_label": 1340}
```

## Usage
**train**

```bash
nsml run -e main.py -d 17_tcls_intent -g 1 -a "--embed_dim 256 --hidden_dim 512 --batch_size 256" 
```

**list saved checkpoint**

```bash
nsml model ls SESSION
```

**submit**

```bash
nsml submit SESSION CHECKPOINT
```
