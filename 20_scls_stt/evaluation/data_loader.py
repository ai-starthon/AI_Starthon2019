import os
import torch
import librosa
from nsml.constants import DATASET_PATH

def get_feature(filepath, feature_size=40):
    sample_rate = 16000
    hop_length = 128

    sig, sample_rate = librosa.core.load(filepath, sample_rate)

    assert sample_rate == 16000,  '%s sample rate must be 16000 but sample-rate is %d' % (filepath, rate)
    assert sig.shape[0] >= 15984, '%s length must be longer than 1 second (frames: %d)' % (filepath, sig.shape[0])

    feat = librosa.feature.mfcc(y=sig, sr=sample_rate, hop_length=hop_length, n_mfcc=feature_size, n_fft=512)
    feat = torch.FloatTensor(mfcc_feat).transpose(0, 1)

    return feat

def feed_infer(output_file, infer_func):

    filepath = os.path.join(DATASET_PATH, 'test', 'test_data', 'test_list.csv')

    with open(output_file, 'w') as of:

        with open(filepath, 'r') as f:

            for no, line in enumerate(f):

                if no >= 3000:
                    break

                # line : "KsponSpeech_617166.wav"

                wav_path = line.strip()
                wav_path = os.path.join(DATASET_PATH, 'test', 'test_data', wav_path)
                pred = infer_func(wav_path)

                of.write('%s,%s\n' % (wav_path, pred))
