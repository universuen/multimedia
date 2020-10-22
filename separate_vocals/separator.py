from spleeter.separator import Separator
from spleeter.audio.adapter import get_default_audio_adapter

separator = Separator('spleeter:2stems')

audio_loader = get_default_audio_adapter()

waveform, _ = audio_loader.load('')