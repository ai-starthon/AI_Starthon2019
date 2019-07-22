import re
import html
import string
import numpy as np

PAD = "[PAD]"
PAD_IND = 0


def get_all_chars():
    koreans = [chr(i) for i in range(44032, 55204)] # 가-힣
    korean_chars = [chr(i) for i in range(ord("ㄱ"), ord("ㅣ") + 1)] # ㄱ-ㅎ, ㅏ-ㅣ
    alphabets = list(string.ascii_letters)
    digits = list(string.digits)
    return [PAD, " "] + koreans + korean_chars + alphabets + digits


# build char vocabulary
vocabs = get_all_chars()
ind2vocab = {ind: char for ind, char in enumerate(vocabs)}
vocab2ind = {char: ind for ind, char in enumerate(vocabs)}

_vocabs = "[^" + "".join(vocabs[1:]) + "]"


def prepro_text(text):
    text = html.unescape(text)
    # text = "".join([char if char in vocabs else " " for char in text])
    text = re.sub(_vocabs, " ", text)
    return re.sub("\s+", " ", text).strip()


def text2ind(text, max_len, raw_text=False):
    if raw_text:
        text = prepro_text(text)
    return np.asarray(list(map(lambda char: vocab2ind[char], text))[:max_len] + \
                      [vocab2ind[PAD] for _ in range(max((max_len - len(text)), 0))])


def ind2text(inds):
    return "".join(map(lambda ind: ind2vocab[ind] if ind >= 0 else "", inds))