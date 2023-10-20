import numpy as np
import librosa
import os, copy
from scipy import signal
import hyperparams as hp
import torch as t

# 解析音频文件，从音频文件中提取 mel 和 mag
def get_spectrograms(fpath):
    # 加载音频文件
    y, sr = librosa.load(fpath, sr=hp.sr)
    # Trimming
    y, _ = librosa.effects.trim(y)
    # 预加重
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
    # 短时傅里叶变换
    linear = librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
    # 幅度谱图
    mag = np.abs(linear)
    # mel 谱图
    mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)
    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    # 规范化
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    # 转置
    mel = mel.T.astype(np.float32)
    mag = mag.T.astype(np.float32)  

    return mel, mag

def spectrogram2wav(mag):
    # 转置
    mag = mag.T
    # 去规范化
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
    # to amplitude
    mag = np.power(10.0, mag * 0.05)
    # 重构音频文件
    wav = griffin_lim(mag**hp.power)
    # 去预加重
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
    # trim
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)
    return y

def invert_spectrogram(spectrogram):
    return librosa.istft(spectrogram, hop_length=hp.hop_length, win_length=hp.win_length, window="hann")

def get_positional_table(d_pos_vec, n_position=1024):
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return t.from_numpy(position_enc).type(t.FloatTensor)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ### Sinusoid position encoding table正弦信号位置编码表
    # param n_position:序列的长度
    # param d_hid:编码后序列中每个位置上数值的尺寸
    # param padding_idx:指定需要padding的位置的索引，即index

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return t.FloatTensor(sinusoid_table)

def guided_attention(N, T, g=0.2):
    W = np.zeros((N, T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(T) - n_pos / float(N)) ** 2 / (2 * g * g))
    return W
