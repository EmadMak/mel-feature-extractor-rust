import ctypes
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
import torch

lib_path = "../rust_env/mel_feature_extractor_rust/target/release/libmel_feature_extractor_rust.so"
audio_file_path = "001001.wav"
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="arabic")

class MelSpectrogramData(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                ("n_frames", ctypes.c_size_t),
                ("n_mels", ctypes.c_size_t)]


try:
    rust_lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Error loading library at {lib_path}: {e}")
    exit()

rust_lib.extract_whisper_features.argtypes = [ctypes.c_char_p]
rust_lib.extract_whisper_features.restype = MelSpectrogramData

rust_lib.free_spectrogram_data.argtypes = [MelSpectrogramData]
rust_lib.free_spectrogram_data.restype = None

audio_path_bytes = audio_file_path.encode("utf-8")

result_struct = rust_lib.extract_whisper_features(audio_path_bytes)

spectrogram = None

if not result_struct.data:
    print("Rust function failed (returned null pointer). Check rust logs.")
else:
    print(f"Received spectrogram: {result_struct.n_mels} mels x {result_struct.n_frames} frames")
    
    try: 
        total_elements = result_struct.n_frames * result_struct.n_mels
        flat_array = np.ctypeslib.as_array(result_struct.data, shape=(total_elements,))

        spectrogram = flat_array.reshape((result_struct.n_mels, result_struct.n_frames)).copy()

        if spectrogram is not None:
            print("Successfully created NumPy array.")
            print("Shape:", spectrogram.shape)
        else:
            print("Couldn't create NumPy array.")
    finally:
        print("Freeing rust memory...")
        rust_lib.free_spectrogram_data(result_struct)
        print("Memory freed.")

if spectrogram is not None:
    print("Spectrogram (copy) available for use.")

input_tensor = torch.tensor(spectrogram).unsqueeze(0)

generated_ids = model.generate(input_features=input_tensor)
text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(text)




