#writefile dataset.py
# wujian@2018

import random
import torch as th
import numpy as np
from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat
from audio import WaveReader

def make_dataloader(train=True, data_kwargs=None, num_workers=4, chunk_size=32000, batch_size=4):
    dataset = Dataset(train=train, **data_kwargs)
    print('dataset length: ', len(dataset))
    if len(dataset) > 18:
        print(f'dataset[18] is : ', dataset[18]['mix'].shape)
    return DataLoader(dataset, train=train, chunk_size=chunk_size, batch_size=batch_size, num_workers=num_workers)

class Dataset(object):
    def __init__(self, mix_scp="", ref_scp=None, sample_rate=8000, train=False):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = [WaveReader(ref, sample_rate=sample_rate) for ref in ref_scp]
        self.index_keys = self.mix.index_keys.copy()
        
        # ★ 固定隨機種子，確保對照實驗抽到一樣的音檔
        random.seed(42)  
        random.shuffle(self.index_keys)
        
        # ★ 將資料量提升至 1/2
        keep_size = max(1, len(self.index_keys) // 2) 
        self.index_keys = self.index_keys[:keep_size]
        
        mode = "訓練 (Train)" if train else "驗證 (Dev)"
        print(f"🚀 [1/2 測試模式] {mode}資料已隨機縮減至 {keep_size} 筆音檔 🚀")

    def __len__(self): return len(self.index_keys)

    def __getitem__(self, index):
        key = self.index_keys[index]
        mix = self.mix[key]
        ref = [reader[key] for reader in self.ref]
        return {"mix": mix.astype(np.float32), "ref": [r.astype(np.float32) for r in ref]}

class ChunkSplitter(object):
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train
    def _make_chunk(self, eg, s):
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = [ref[s:s + self.chunk_size] for ref in eg["ref"]]
        return chunk
    def split(self, eg):
        N = eg["mix"].size
        if N < self.least: return []
        chunks = []
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = [np.pad(ref, (0, P), "constant") for ref in eg["ref"]]
            chunks.append(chunk)
        else:
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N: break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks

class DataLoader(object):
    def __init__(self, dataset, num_workers=4, chunk_size=32000, batch_size=16, train=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size, train=train, least=chunk_size // 2)
        # ★ 加入 pin_memory=True
        self.eg_loader = dat.DataLoader(dataset, batch_size=batch_size // 2, num_workers=num_workers, shuffle=train, collate_fn=self._collate, pin_memory=True)
    def _collate(self, batch):
        chunk = []
        for eg in batch: chunk += self.splitter.split(eg)
        return chunk
    def _merge(self, chunk_list):
        N = len(chunk_list)
        if self.train: random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(chunk_list[s:s + self.batch_size])
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []
    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch: yield obj