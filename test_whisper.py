import os
os.add_dll_directory(r"C:\Users\smkja\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\nvidia\cublas\bin")
os.add_dll_directory(r"C:\Users\smkja\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\nvidia\cudnn\bin")
from faster_whisper import WhisperModel
import time

# Start with "small" model — good accuracy, light on VRAM (~1GB)
print("Loading Whisper model...")
start = time.time()
model = WhisperModel("small", device="cuda", compute_type="float16")
print(f"Model loaded in {time.time() - start:.1f}s")

# Transcribe
audio_path = r"C:\Users\smkja\Recording.m4a"  # update this path
print(f"Transcribing {audio_path}...")

start = time.time()
segments, info = model.transcribe(audio_path, beam_size=5)

print(f"\nLanguage: {info.language} (probability: {info.language_probability:.2f})")
print(f"Duration: {info.duration:.1f}s\n")
print("--- Transcript ---")

full_text = []
for segment in segments:
    print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")
    full_text.append(segment.text)

elapsed = time.time() - start
print(f"\n--- Done in {elapsed:.1f}s ---")
print(f"Real-time factor: {elapsed/info.duration:.2f}x (lower = faster)")

# Save transcript for tomorrow
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(full_text))
print("Transcript saved to transcript.txt")