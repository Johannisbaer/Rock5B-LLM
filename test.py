import sounddevice as sd
import scipy.io.wavfile as wav
import subprocess
import time
import requests
import json
import os
import random
import simpleaudio as sa
import soundfile as sf
import pygame
from scipy.signal import resample
import numpy as np

# === CONFIG ===
TEMP_DIR = "temp"
DURATION = 7
SAMPLE_RATE = 14000
DEVICE = None  # Use default input

WHISPER_MODEL = "/home/localai/whisper.cpp/models/ggml-base.en.bin"
WHISPER_BIN = "/home/localai/whisper.cpp/build/bin/whisper-cli"

LLM_API = "http://127.0.0.1:8080/api/chat"
LLM_MODEL = "Qwen3-1.7B-1.2.0"
MAX_TOKENS = 256
TEMPERATURE = 0.7

PAROLI_SYNTH = "http://127.0.0.1:8848/api/v1/synthesise"
PAROLI_SPEAKER_ID = None

play_obj = None
pygame.mixer.pre_init(frequency=16000, size=-16, channels=1, buffer=4096)
pygame.init()

def control_music(action, filepath=None):
    global play_obj
    if action == "play":
        if filepath is None:
            print("You must provide a file to play.")
            return
        wave_obj = sa.WaveObject.from_wave_file(filepath)
        play_obj = wave_obj.play()
    elif action == "stop":
        if play_obj is not None:
            play_obj.stop()
            play_obj = None
    else:
        print(f"Unknown action: {action}")

def record_chunk(filename):
    print(f"🎙️ Recording {DURATION}s…")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16",
                   device=DEVICE)
    sd.wait()
    wav.write(filename, SAMPLE_RATE, audio)
    print(f"💾 Saved {filename}")


def transcribe_chunk(filename):
    control_music("play", r"music/quiz-game-show-quiet.wav")
    print(f"📝 Transcribing {filename}…")
    cmd = [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", filename, "-otxt"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    text = result.stdout.strip()
    print(f"📜 Transcript: {text!r}")

    output_txt = filename + ".txt"
    try:
        os.remove(output_txt)
        print(f"🗑️ Deleted {output_txt}")
    except Exception as e:
        print(f"⚠️ Could not delete the file {output_txt}. Error: {e}")

    return text


def call_llm(
    prompt,
    system_prompt="You are a helpful assistant for teenagers. You have no internet connection at all, so you cannot answer on weather or local events. You answer in 3-4 short sentences.",
    enable_thinking=True,
    temperature=0.7,
    max_tokens=256
):

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    print("\n🤖 LLM RESPONSE:\n")

    response_part = ""
    try:
        response = requests.post(LLM_API, json=payload)
        response.raise_for_status()

        # Extract the message directly
        json_data = response.json()
        if "message" in json_data:
            full_response = json_data["message"]["content"]

            # Extract text after </think>
            think_tag = "</think>"
            start_position = full_response.find(think_tag)
            if start_position != -1:
                response_part = full_response[start_position + len(think_tag):].strip()
                print(response_part)

                # Send this part to TTS
                if response_part:
                    speak_with_http(response_part)
            else:
                print("No </think> tag found. Full response is:", full_response)

    except requests.exceptions.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        with open("history.txt", "a") as f:
            f.write(f"Prompt: {prompt}\n")
            if response_part:
                f.write(f"Response: {response_part}\n\n")

    print("\n✅ Done.\n")

def pitch_shift(data, sr, semitones):
    factor = 2 ** (semitones / 12)  # semitone ratio
    new_len = int(len(data) / factor)
    resampled = resample(data, new_len)
    return resampled.astype(np.int16)


def speak_with_http(text):
    safe_text = text.strip()
    if len(safe_text) < 5:
        print("🛑 TTS aborted: text too short.")
        return

    payload = {
        "text": "... ... " + safe_text + " ... ...",
        "audio_format": "opus"
    }
    if PAROLI_SPEAKER_ID:
        payload["speaker_id"] = PAROLI_SPEAKER_ID

    print(f"\n🔈 Sending to Paroli: {payload}")
    r = requests.post(PAROLI_SYNTH, json=payload)
    if not r.ok:
        print(f"❌ Paroli error: {r.status_code} {r.text}")
        return

    ogg_file = "paroli_tts.ogg"
    with open(ogg_file, "wb") as f:
        f.write(r.content)

    control_music("stop")

    try:
        data, samplerate = sf.read(ogg_file, dtype='int16')
        channels = 1 if data.ndim == 1 else data.shape[1]

        # Pitch shift by semitones (change this!)
            # +4	Higher pitch
            # -4	Lower pitch
            # +7	Chipmunk voice
            # -7	Deep/robotic
        semitones = +4
        if channels == 1:
            data = pitch_shift(data, samplerate, semitones)
        else:
            # pitch-shift each channel separately
            data = np.column_stack([
                pitch_shift(data[:, 0], samplerate, semitones),
                pitch_shift(data[:, 1], samplerate, semitones)
            ])

        play_obj = sa.play_buffer(data.tobytes(), channels, 2, samplerate)
        play_obj.wait_done()
        control_music("stop")
    except Exception as e:
        control_music("stop")
        print(f"⚠️ Playback failed: {e}")



def main():
    count = 0
    print("▶️ Audio → Whisper → Qwen3 (rkllama /chat) loop starting…")
    try:
        while True:
            wav_file = os.path.join(TEMP_DIR, f"chunk_{count}.wav")
            record_chunk(wav_file)
            transcript = transcribe_chunk(wav_file)
            if transcript and "BLANK_AUDIO" not in transcript:
                call_llm(transcript)
            else:
                print("⚠️ No speech detected or blank audio.")
            count += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")


if __name__ == "__main__":
    main()
