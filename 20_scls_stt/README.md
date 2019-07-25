# nipa-speech-2019

NIPA Speech Recognition Baseline

### 실행
* ./run.sh [ENTER]
* run.sh
   ```
   #!/bin/sh

   BATCH_SIZE=64
   WORKER_SIZE=4
   GPU_SIZE=2
   CPU_SIZE=4

   nsml run -d 20_scls_stt -g $GPU_SIZE -c $CPU_SIZE -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention"
   ```

### files
* script.labels
   * 정답 낱글자들에 대한 레이블 매핑 테이블
   * "_" : padding
   * "\<s\>" : begin of sentence
   * "\</s\>" : end of sentence
* train/train_label
   * 모드 training wav파일들의 정답 레이블 리스트 저장
   * 정답 레이블의 실제 문자열은 sript.labels와 매칭해서 확이 가능
   ```
   # <file-name>,<label list>
   KsponSpeech_000164,275 46 1012 1155 1021 1155 1899 1300 1155 904 497 1155 
   KsponSpeech_000169,743 1155 497 1608 497 1155 857 1527 1155 1783 226 1155 541 1155
   ...
   ```
* train/train_data/data_list.csv
   * 11만개의 학습 wav파일 리스트 저장
   ```
   # <wav-filename>,<script-filename>
   KsponSpeech_539961.wav,KsponSpeech_539961.label
   KsponSpeech_616116.wav,KsponSpeech_616116.label
   ...
   ```
### evaluation
* 공백 문자는 제거하고 CER을 계산함
   * "오늘 학교에 갔다" => "오늘학교에갔다"
* 최종 출력값은 Character Recoginition Rate
   * CRR = (1.0 - CER (character error rate)) * 100.0
      * 100.0 : 100% 정답과 일치 (공백문자 제외)

### options

```
usage: main.py [-h] [--feature_size FEATURE_SIZE] [--hidden_size HIDDEN_SIZE]
               [--layer_size LAYER_SIZE] [--dropout DROPOUT] [--bidirectional]
               [--use_attention] [--batch_size BATCH_SIZE] [--workers WORKERS]
               [--epochs EPOCHS] [--lr LR] [--teacher_forcing TEACHER_FORCING]
               [--max_len MAX_LEN] [--no_cuda] [--seed SEED]
               [--save_name SAVE_NAME] [--mode MODE] [--pause PAUSE]

NIPA Speech Recognition Baseline

optional arguments:
  -h, --help            show this help message and exit
  --feature_size FEATURE_SIZE
                        size of MFCC feature (default: 40)
  --hidden_size HIDDEN_SIZE
                        hidden size of model (default: 256)
  --layer_size LAYER_SIZE
                        number of layers of model (default: 3)
  --dropout DROPOUT     dropout rate in training (default: 0.2)
  --bidirectional       use bidirectional RNN for encoder (default: False)
  --use_attention       use attention between encoder-decoder (default: False)
  --batch_size BATCH_SIZE
                        batch size in training (default: 32)
  --workers WORKERS     number of workers in dataset loader (default: 4)
  --epochs EPOCHS       number of epochs in training (default: 100)
  --lr LR               learning rate (default: 0.0001)
  --teacher_forcing TEACHER_FORCING
                        teacher forcing ratio in decoder (default: 0.5)
  --max_len MAX_LEN     maximum characters of sentence (default: 80)
  --no_cuda             disables CUDA training
  --seed SEED           random seed (default: 1)
  --save_name SAVE_NAME
                        the model name for saving in nsml (default is 'model', best-model name is 'best')
  --mode MODE
  --pause PAUSE
  ```
### Reference
* pytorch-seq2seq (https://github.com/IBM/pytorch-seq2seq) 의 seq2seq 모델을 사용
* NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE (https://arxiv.org/pdf/1409.0473.pdf)
