import glob
import numpy as np
import os
import random
import tensorflow as tf
import tqdm
import csv
import numpy as np


def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        elif path.endswith('.csv'):
            start_token = "<|startoftext|>"
            end_token = "<|endoftext|>"
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                fp.readline()   # skip header
                reader = csv.reader(fp)
                for row in reader:
                    raw_text += start_token + row[0] + end_token + "\n"
        else:
            # Plain text
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi

class FileSampler(object):

    def __init__(self, enc, path, length=None, eager=True):
        self._total_size = None
        self._encode_path = None

        self.path = path
        self.enc = enc
        self.length = length




    def encode_file():
        n_toks = 0
        pbar = tqdm.tqdm()
        with open(self.path, 'r') as in_f, open(self.encode_path, 'wb') as out_f:
            line = in_f.readline()
            tokens = enc.encode(line)
            for t in tokens:
                out_f.write(t)
            pbar.update()
        pbar.close()

    def sample(self, length=None):
        if not length:
            length = self.length

        assert(length)

        ind = np.random.randint(0, self.total_size-length)

        with open(self.encode_path, 'rb') as enc_f:
            enc_f.seek(ind)
            return enc_f.read(length)




    @property
    def total_size(self):
        if not self._total_size:
            with open(self.encode_path, 'rb') as ofile:
                ofile.seek(0, 2)
                self._total_size = ofile.tell()
        return self._total_size


    @property
    def encode_path(self):
        if not self._encode_path:
            root, ext = os.path.splitext(self.path)
            self._encode_path = root + '_encoded.txt'

            if not os.path.exists(self._encode_path):
                print('encoding to file')
                self.encode_file()

        return self._encode_path



class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = random.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]
