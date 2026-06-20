from tokenize import group
import torch as th
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as F

def param(nnet, Mb=True):
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles

class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)
    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(self.__name__))
        x = th.transpose(x, 1, 2)
        x = super().forward(x)
        x = th.transpose(x, 1, 2)
        return x

class GlobalChannelLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(self.__name__))
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

def build_norm(norm, dim):
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)

class Conv1D(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)
    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze: x = th.squeeze(x)
        return x

class ConvTrans1D(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)
    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze: x = th.squeeze(x)
        return x

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels=256, conv_channels=512, groups=16, Sc=512, kernel_size=3, dilation=1, norm="cLN", causal=False):
        super(Conv1DBlock, self).__init__()
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        groups = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        
        groupoutchnl = 256
        self.shuffgroupconv = nn.Conv1d(conv_channels, groupoutchnl, kernel_size, groups=groups, padding=dconv_pad, dilation=dilation, bias=True)
        self.tanh1 = nn.Tanh()
        self.dconv = nn.Conv1d(conv_channels, conv_channels, kernel_size, groups=conv_channels, padding=dconv_pad, dilation=dilation, bias=True)
        self.shuffgroup = ChannelShuffle(conv_channels, groups)
        self.conv1x1_2 = Conv1D(conv_channels, groupoutchnl, 1)    
        self.sigmoid1 = nn.Sigmoid()   
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, groupoutchnl * 2)
        self.sconv = nn.Conv1d(groupoutchnl * 2, in_channels, 1, bias=True)
        self.skip_out = nn.Conv1d(groupoutchnl * 2, Sc, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        sh = self.shuffgroup(y)
        shuffcov = self.shuffgroupconv(sh)
        shufftan = self.tanh1(shuffcov)
        shuffsigm = self.sigmoid1(shuffcov)
       
        y = self.dconv(y)
        y = self.conv1x1_2(y)
        depsigm = self.sigmoid1(y)
        deptan = self.tanh1(y)
        _x_up = shufftan * depsigm
        _x_down = shuffsigm * deptan
        y = th.cat((_x_up, _x_down), axis=1)
        
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        out = self.sconv(y)
        skip = self.skip_out(y)
        x = x + out
        return skip, x

def channel_shuffleforsound(x, groups):
    batch, inchannel, height = x.size()
    channels_per_group = inchannel // groups
    x = x.view(batch, channels_per_group, groups, height)
    x = th.transpose(x, 1, 2).contiguous()
    x = x.view(batch, inchannel, height)
    return x

class ChannelShuffle(nn.Module):
    def __init__(self, channels, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        return channel_shuffleforsound(x, self.groups)

# =======================================================
# 👑 方向B: Temporal Gate 主模型
# =======================================================
class MS_SL2_split_model(nn.Module):
    def __init__(self, L=16, N=512, X=8, R=2, B=256, Sc=256, Slice=2, H=512, P=3, norm="gLN", num_spks=2, non_linear="sigmoid", causal=False):
        super(MS_SL2_split_model, self).__init__()
        supported_nonlinear = {"relu": F.relu, "sigmoid": th.sigmoid, "softmax": F.softmax}
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}", format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        
        self.encoder_1d = Conv1D(1, N, L, stride=L // 2, padding=0)
        self.a = nn.Conv1d(in_channels=N, out_channels=257, kernel_size=1)
        self.ln = ChannelWiseLayerNorm(N)
        self.proj = Conv1D(N, B, 1)
        
        self.slices = self._build_slices(Slice, R, X, Sc=Sc, in_channels=B, conv_channels=H, kernel_size=P, norm=norm, causal=causal)
            
        max_val = 0.07331312
        min_val = -0.0814228
        self.wList = nn.Parameter((max_val - min_val) * th.rand(4) + min_val, requires_grad=True) 
        
        # ★ 方向B 核心: 為每一個 Slice 建立一個 1x1 卷積層，用來沿著時間軸學出 Gate
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(in_channels=Sc, out_channels=Sc, kernel_size=1) 
            for _ in range(Slice)
        ])
        
        self.PRelu = nn.PReLU()
        self.mask = Conv1D(Sc, num_spks * N, 1)
        self.decoder_1d = ConvTrans1D(N, 1, kernel_size=L, stride=L // 2, bias=True)
        self.num_spks = num_spks
        self.R = R 
        self.X = X 
        self.slice = Slice 
    
    def _build_blocks(self, num_blocks, **block_kwargs):
        blocks = [Conv1DBlock(**block_kwargs, dilation=(2**b)) for b in range(num_blocks)]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        repeats = [self._build_blocks(num_blocks, **block_kwargs) for r in range(num_repeats)]
        return nn.Sequential(*repeats)
    
    def _build_slices(self, num_slice, num_repeats, num_blocks, **block_kwargs):
        slices = [self._build_repeats(num_repeats, num_blocks, **block_kwargs) for r in range(num_slice)]
        return nn.Sequential(*slices)

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError("{} accept 1/2D tensor as input, but got {:d}".format(self.__name__, x.dim()))
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
        
        w = self.encoder_1d(x)
        w = th.unsqueeze(w, 2)
        w = th.squeeze(w, 2)
        w = w[:, :256, :] 
        out = th.stft(x, n_fft=512, hop_length=8, win_length=64, return_complex=True)
        out = out.real
        out = out[:, :256, :-2] 
        w = th.cat((w, out), 1)
        
        w = self.ln(w)
        y = self.proj(w)
        
        skip_connection = 0
        Slice_input = y
        Slices_Output = 0

        for Slice in range(self.slice):
            for i in range(self.R):
                for j in range(self.X):
                    skip, y = self.slices[Slice][i][j](y)
                    skip_connection = skip_connection + skip
                    
            # ★ 方向B 核心實作: 用 1x1 卷積 + Sigmoid 算出時間閘門 gate
            gate = th.sigmoid(self.temporal_convs[Slice](skip_connection))
            
            # 保留原本 wList 算出的整體重要性 (維持對照實驗的基準一致)
            w_sum = self.wList.sum()
            
            # 結合時間閘門進行加權，這能完美控管「哪些 Time Steps 要啟用」
            slice_row_sum = skip_connection * w_sum * gate
            
            if Slice == 0:
                Slices_Output = slice_row_sum
            else:
                Slices_Output = th.add(Slices_Output, slice_row_sum)
             
            skip_connection = 0
            y = Slice_input
        
        y = self.PRelu(Slices_Output)
        e = th.chunk(self.mask(y), self.num_spks, 1)
           
        if self.non_linear_type == "softmax":
            m = self.non_linear(th.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(th.stack(e, dim=0))
        
        s = [w * m[n] for n in range(self.num_spks)]
        return [self.decoder_1d(x, squeeze=True) for x in s]