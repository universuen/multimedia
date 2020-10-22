# from pydub import AudioSegment
# import matplotlib.pyplot as plt
# import numpy as np
#
# class KeyDetector:
#     def __init__(self, wav_path, debug=False):
#         """
#         初始化
#         :param wav_path:.wav文件存储路径
#         """
#         # 读取音频
#         music = AudioSegment.from_wav(wav_path)
#         # 过滤低频信息
#         self.music = music.high_pass_filter(80)
#         # 以50ms为采样宽度，获取响度序列
#         self.segment_ms = 50
#         self.volume = [seg.dBFS for seg in self.music[::self.segment_ms]]
#         if debug:
#             x_axis = np.arange(len(self.volume)) * (self.segment_ms / 1000)
#             plt.xlabel("time (s)")
#             plt.ylabel("volume (dB)")
#             plt.plot(x_axis, self.volume)
#             plt.show()
#         self._detect_taps()
#
#     # 识别击键时间点(s)
#     def _detect_taps(self):
#         # 最短击键间隔(ms)
#         min_period_ms = 100
#         # 最小音量(dB)
#         volume_threshold_db = -35
#         # 最小跳跃间隔(dB)
#         edge_threshold_db = 5
#         self.taps = list()
#         for i in range(1, len(self.volume)):
#             # 足够响并且足够突然
#             if self.volume[i] > volume_threshold_db and self.volume[i] - self.volume[i-1] > edge_threshold_db:
#                 tap_time = i*self.segment_ms
#                 # 间隔足够长
#                 if len(self.taps) == 0 or tap_time - self.taps[-1] >= min_period_ms:
#                     self.taps.append(tap_time/1000)
#
#     # 获取每一段样本的最大频率
#     def _get_frequency(self):
#         self.freq_array = list()
#         for i, tap_time in enumerate(self.taps):
#             sample_from = tap_time + 50
#             sample_to = tap_time + 550
#             if i < len(self.taps) - 1:
#                 sample_to = min(self.taps[i + 1], sample_to)
#             segment = self.music[sample_from:sample_to]
#             freqs, freq_magnitudes = frequency_spectrum(segment)
#
# if __name__ == '__main__':
#     kd = KeyDetector("summer.wav", debug=True)
#     print(kd.taps)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from note_detection.key2freq import KEY2FREQ


class KeyDetector:
    def __init__(self, wav_path):
        # 获取每个琴键的频率
        self.key2freq = KEY2FREQ
        # 读取音频
        print("正在读取音频")
        self.music, sr = librosa.load(wav_path, sr=8000)
        # 短时傅里叶变换获取每一段最大频率
        print("正在计算频谱")
        self.fft = librosa.stft(self.music)
        self.freq = np.abs(self.fft)
        self.max_freq = list()
        for i in self.freq:
            self.max_freq.append(np.where(i == np.max(i))[0])

    def generate_keys(self):
        result = list()
        # 为每一个频率匹配频率最接近的琴键
        print("正在根据频谱匹配琴键")
        for music_freq in self.max_freq:
            freq_abs = music_freq
            final_key = None
            for key, value in self.key2freq.items():
                if abs(value - music_freq) < freq_abs:
                    freq_abs = abs(value - music_freq)
                    final_key = key
            result.append(final_key)
        print("匹配完成")
        return result


if __name__ == '__main__':
    kd = KeyDetector("钢琴音频.wav")
    keys = kd.generate_keys()
    # 保存结果
    with open("keys.txt", "w") as f:
        f.write(", ".join(keys))
