import os

from pydub import AudioSegment
import subprocess
from app.pipeline.config import CHUNK_LENGTH, CHUNK_OVERLAP

def cut_segment(chunk_file:str, segment_file:str, start, end, segment_index):
    # Cut the relevant segment with ffmpeg
    command = [
        "ffmpeg", "-y",
        "-i", chunk_file,
        "-ss", str(start),
        "-to", str(end),
        "-acodec", "copy",
        segment_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"FFmpeg failed for segment {segment_index}: {e}")

def convert_mp4_to_wav(input_path: str, output_path: str) -> None:
    """
    Converts an MP4 video file to a 16kHz mono WAV file using ffmpeg.

    Args:
        input_path (str): Path to input .mp4 file
        output_path (str): Path where output .wav file will be saved
    Raises:
        RuntimeError: If ffmpeg fails
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz
        "-y",                # overwrite output file if exists
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e}")

def split_wav_into_chunks(wav_path:str, chunk_dir:str, chunk_length = CHUNK_LENGTH) -> None:
    """
        Splits a WAV file into fixed-length chunks (default: 60 sec).
    """
    def to_ms(val):
        return val * 1000
    def to_s(val):
        return val / 1000
    os.makedirs(chunk_dir, exist_ok=True)

    audio = AudioSegment.from_wav(wav_path)
    duration_sec = to_s(len(audio))
    chunk_count = int(duration_sec // chunk_length) + 1

    for i in range(chunk_count):
        start_ms = to_ms(i * chunk_length)
        end_ms = min(to_ms((i+1) * chunk_length), len(audio))
        chunk = audio[start_ms:end_ms]
        chunk_path = os.path.join(chunk_dir, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")
        print(f"Exported {chunk_path} ({round(to_s((end_ms - start_ms)), 2)}s)")

import os
from typing import List, Optional
from pydub import AudioSegment

def split_wav_into_chunks_v2(
    wav_path: str,
    chunk_dir: str,
    chunk_length: float = CHUNK_LENGTH,          # saniye
    overlap: float = CHUNK_OVERLAP,         # saniye
    force_mono: bool = True,      # True ise mono'ya downmix eder
    target_rate: Optional[int] = None  # örn. 16000; None ise olduğu gibi bırakır
) -> List[str]:
    """
    WAV dosyasını sabit uzunlukta ve overlap'lı parçalara böler.
    Örn: chunk_length=240, overlap=2 → 240 sn pencereler, 238 sn hop.

    Dönüş:
        Parça dosya yollarının kronolojik listesi.
    """
    if overlap >= chunk_length:
        raise ValueError("overlap, chunk_length değerinden küçük olmalıdır.")

    os.makedirs(chunk_dir, exist_ok=True)

    audio = AudioSegment.from_wav(wav_path)

    if force_mono and audio.channels != 1:
        audio = audio.set_channels(1)

    if target_rate is not None and audio.frame_rate != target_rate:
        audio = audio.set_frame_rate(target_rate)

    win_ms = int(chunk_length * 1000.0)
    hop_ms = int((chunk_length - overlap) * 1000.0)
    duration_ms = len(audio)

    paths: List[str] = []
    i = 0
    start_ms = 0

    while start_ms < duration_ms:
        end_ms = min(start_ms + win_ms, duration_ms)
        chunk = audio[start_ms:end_ms]

        out_path = os.path.join(chunk_dir, f"chunk_{i:04d}.wav")
        chunk.export(out_path, format="wav")
        paths.append(out_path)

        print(f"Exported {out_path} ({(end_ms - start_ms)/1000.0:.2f}s)")

        if end_ms >= duration_ms:
            break

        start_ms += hop_ms
        i += 1

    return paths



