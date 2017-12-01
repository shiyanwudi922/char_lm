import numpy as np
import copy
import pickle

def batch_generator(int_arr, batch_size, max_time):
    arr = copy.copy(int_arr)

    arr_len = len(arr)
    batch_cnt = int((int(arr_len/batch_size)-1)/max_time)
    arr = arr[:batch_size*(max_time*batch_cnt+1)]
    arr = arr.reshape((batch_size, -1))

    print("arr shape: " + str(arr.shape))
    print("batch size: " + str(batch_size))
    print("max time: " + str(max_time))
    print("batch cnt: " + str(batch_cnt))

    while True:
        np.random.shuffle(arr)
        for n in range(0, batch_cnt):
            x = arr[:, n*max_time:(n+1)*max_time]
            y = arr[:, n*max_time+1:(n+1)*max_time+1]
            yield x, y


class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, vocab_file=None):
        if vocab_file is not None:
            with open(vocab_file, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print("origin vocab size of the text is: " + str(len(vocab)))
            # max_vocab_process
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            # vocab_count_list = []
            # for word in vocab_count:
            #     vocab_count_list.append((word, vocab_count[word]))
            # vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            vocab_count_list = sorted(vocab_count, key=vocab_count.get, reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.ch2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2ch = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def ch_to_idx(self, ch):
        if ch in self.ch2idx:
            return self.ch2idx[ch]
        else:
            return len(self.vocab)

    def idx_to_ch(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.idx2ch[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for ch in text:
            arr.append(self.ch_to_idx(ch))
        return np.array(arr)

    def arr_to_text(self, arr):
        chars = []
        for index in arr:
            chars.append(self.idx_to_ch(index))
        return "".join(chars)

    def save_to_file(self, vocab_file):
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.vocab, f)
