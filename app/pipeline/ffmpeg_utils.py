import os
from typing import List, Optional

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
    print(f"Converting MP4 to WAV: {input_path} -> {output_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        raise RuntimeError(f"Input file does not exist: {input_path}")
    
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz
        "-y",                # overwrite output file if exists
        output_path
    ]

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"FFmpeg conversion successful. Output file size: {os.path.getsize(output_path) if os.path.exists(output_path) else 'file not found'} bytes")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg stderr: {e.stderr}")
        print(f"FFmpeg stdout: {e.stdout}")
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
    
    print(f"Audio properties: duration={len(audio)/1000.0:.2f}s, channels={audio.channels}, frame_rate={audio.frame_rate}")

    if force_mono and audio.channels != 1:
        audio = audio.set_channels(1)
        print("Converted to mono")

    if target_rate is not None and audio.frame_rate != target_rate:
        audio = audio.set_frame_rate(target_rate)
        print(f"Resampled to {target_rate} Hz")

    win_ms = int(chunk_length * 1000.0)
    hop_ms = int((chunk_length - overlap) * 1000.0)
    duration_ms = len(audio)
    
    print(f"Chunking parameters: chunk_length={chunk_length}s, overlap={overlap}s, window={win_ms}ms, hop={hop_ms}ms")
    print(f"Audio duration: {duration_ms}ms ({duration_ms/1000.0:.2f}s)")
    
    if duration_ms < win_ms:
        print(f"WARNING: Audio duration ({duration_ms/1000.0:.2f}s) is shorter than chunk length ({chunk_length}s)")

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

    print(f"Chunking complete: Created {len(paths)} chunks")
    return paths



