#writefile trainer.py
# -*- coding: utf-8 -*-
# wujian@2018

import os
import sys
import time
import matplotlib.pyplot as plt
from itertools import permutations
from collections import defaultdict
import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from utils import get_logger
from conf import trainer_conf, nnet_conf, train_data, dev_data, chunk_size

def load_obj(obj, device):
    def cuda(obj): return obj.to(device) if isinstance(obj, th.Tensor) else obj
    if isinstance(obj, dict): return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list): return [load_obj(val, device) for val in obj]
    else: return cuda(obj)

class SimpleTimer(object):
    def __init__(self): self.reset()
    def reset(self): self.start = time.time()
    def elapsed(self): return (time.time() - self.start) / 60

class ProgressReporter(object):
    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.timer = SimpleTimer()
    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches (loss = {:+.2f})...".format(N, avg))
    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {"loss": sum(self.loss) / N, "batches": N, "cost": self.timer.elapsed()}

class Trainer(object):
    def __init__(self, nnet, checkpoint="/media/kaldi/SP1/MingHshuan/checkpoint", optimizer="adam",
                 gpuid=0, optimizer_kwargs=None, clip_norm=None, min_lr=0, patience=0, factor=0.5,
                 logging_period=100, resume=None, no_impr=6, loss_mode="snr"):
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        if not th.cuda.is_available(): raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple): gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint): os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(os.path.join(checkpoint, "trainer.log"), file=True)
        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0 
        self.no_impr = no_impr
        self.loss_mode = loss_mode

        if resume:
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            # ★ 必須補上這行，否則接關會當機！
            self.trainable_params = [p for p in self.nnet.parameters() if p.requires_grad]
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            self.trainable_params = [p for p in self.nnet.parameters() if p.requires_grad]
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
            
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr)
        self.num_params = sum([param.nelement() for param in nnet.parameters()]) / 10.0**6

    def save_checkpoint(self, best=True):
        cpt = {"epoch": self.cur_epoch, "model_state_dict": self.nnet.state_dict(), "optim_state_dict": self.optimizer.state_dict()}
        th.save(self.nnet.state_dict(), os.path.join(self.checkpoint, "{}.pt.tar".format("best" if best else self.cur_epoch)))
        th.save(cpt, os.path.join(self.checkpoint, "{}.pt.tar".format("best" if best else self.cur_epoch)))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {"sgd": th.optim.SGD, "rmsprop": th.optim.RMSprop, "adam": th.optim.Adam, 
                               "adadelta": th.optim.Adadelta, "adagrad": th.optim.Adagrad, "adamax": th.optim.Adamax}
        opt = supported_optimizer[optimizer](self.trainable_params, **kwargs)
        if state is not None: opt.load_state_dict(state)
        return opt

    def compute_loss(self, egs): raise NotImplementedError

    def train(self, data_loader):
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)
        for egs in tqdm(data_loader, desc=f"Train Epoch {self.cur_epoch}", leave=True, ncols=100):
            egs = load_obj(egs, self.device)
            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)
            loss.backward()
            if self.clip_norm: clip_grad_norm_(self.trainable_params, self.clip_norm)
            self.optimizer.step()
            reporter.add(loss.item())
        return reporter.report()

    def eval(self, data_loader):
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)
        with th.no_grad():
            for egs in tqdm(data_loader, desc=f"Eval Epoch {self.cur_epoch}", leave=True, ncols=100):
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs)
                reporter.add(loss.item())
        return reporter.report(details=True)

    def run(self, train_loader, dev_loader, num_epochs=50):
        with th.cuda.device(self.gpuid[0]):
            stats = dict()
            self.save_checkpoint(best=False)
            print('\n[系統] 開始初始驗證 (Start Eval)...')
            cv = self.eval(dev_loader)
            best_loss = cv["loss"]
            no_impr = 0
            self.scheduler.best = best_loss
            
            train_loss_lst = []
            dev_loss_lst = []
            
            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                tr = self.train(train_loader)
                cv = self.eval(dev_loader)
                
                train_loss_lst.append(tr['loss'])
                dev_loss_lst.append(cv['loss'])
                
                if cv["loss"] > best_loss:
                    no_impr += 1
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(best=True)
                self.scheduler.step(cv["loss"])
                sys.stdout.flush()
                self.save_checkpoint(best=False)
                if no_impr == self.no_impr: break
            
            # 傳入正確的存檔路徑
            plot(train_loss_lst, dev_loss_lst, len(train_loss_lst), self.checkpoint)

class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs): super(SiSnrTrainer, self).__init__(*args, **kwargs)
    def sisnr(self, x, s, eps=1e-8):
        def l2norm(mat, keepdim=False): return th.norm(mat, dim=-1, keepdim=keepdim)
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    def snr(self, x, s, eps=1e-8):
        def l2norm(mat, keepdim=False): return th.norm(mat, dim=-1, keepdim=keepdim)
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        return 20 * th.log10(eps + l2norm(s_zm) / (l2norm(s_zm - x_zm) + eps))
    def compute_loss(self, egs):
        ests = th.nn.parallel.data_parallel(self.nnet, egs["mix"], device_ids=self.gpuid)
        refs = egs["ref"]
        num_spks = len(refs)
        def sisnr_loss(permute): return sum([self.sisnr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
        def snr_loss(permute): return sum([self.snr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)
        N = egs["mix"].size(0)
        if self.loss_mode == "snr": sisnr_mat = th.stack([snr_loss(range(num_spks))])
        elif self.loss_mode == "sisnr": sisnr_mat = th.stack([sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, _ = th.max(sisnr_mat, dim=0)
        return -th.sum(max_perutt) / N

def plot(train_loss_lst, dev_loss_lst, actual_epochs, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, actual_epochs + 1), train_loss_lst, label='Train', color='red', marker='o')
    plt.plot(range(1, actual_epochs + 1), dev_loss_lst, label='Dev', color='blue', marker='o')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (SI-SNR)')
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(save_dir, 'loss_vs_epoch.png')
    plt.savefig(save_path)
    print(f"\n✅ 訓練曲線圖已成功儲存至: {save_path}")
    plt.close()