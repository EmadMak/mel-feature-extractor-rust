use hound::{WavReader, WavSpec, SampleFormat, WavWriter};
use std::{ffi::CStr, i16};
use std::os::raw::c_char;
use rubato::{SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction, Resampler};

fn read_wav(path: &str) -> Result<(Vec<f32>, u32), String> {
    let reader = WavReader::open(path)
        .map_err(|e| format!("Failed to open WAV: {}", e))?;
    let spec = reader.spec();
    let orig_sample_rate = spec.sample_rate;
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();

    let mono = if spec.channels == 1 {
        samples
    } else {
        samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    };

    Ok((mono, orig_sample_rate))
}

fn resample_audio(input: Vec<f32>, orig_rate: u32, target_rate: u32) -> Result<Vec<f32>, String> {
    if orig_rate == target_rate {
        return Ok(input)
    }

    let ratio = target_rate as f64 / orig_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, input.len(), 1)
        .map_err(|e| format!("Resampler init error: {:?}", e))?;
    let outputs = resampler
        .process(&[input], None)
        .map_err(|e| format!("Resampler init error: {:?}", e))?;
    Ok(outputs.into_iter().next().unwrap())
}

fn pad_or_truncate(input: Vec<f32>, target_len: usize) -> Result<Vec<f32>, String> {
    if input.len() == target_len {
        return Ok(input);
    }
    if input.len() > target_len {
        return Ok(input[..target_len].to_vec());
    }
    let mut padded = input;
    padded.resize(target_len, 0.0);
    Ok(padded)
}

fn save_wav(out_path: &str, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(out_path, spec)
        .map_err(|e| format!("Failed to open WAV: {}", e))?;

    for &sample in samples {
        let scaled = (sample.max(-1.0).min(1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(scaled)
            .map_err(|e| format!("Failed to write sample: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize wav: {}", e))?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn extract_whisper_features(path: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Invalid UTF-8 path");
            return;
        }
    };

    let (mono, orig_sample_rate) = match read_wav(path_str) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{}", err);
            return;
        }
    };

    let resampled = match resample_audio(mono, orig_sample_rate, 16000) {
        Ok(buf) => buf,
        Err(err) => {
            eprintln!("{}", err);
            return;
        }
    };

    let mel_ready = match pad_or_truncate(resampled, 480000) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{}", err);
            return;
        }
    };

    println!("Final sample count: {}", mel_ready.len());

    if let Err(err) = save_wav("resampled.wav", &mel_ready, 16000) {
        eprintln!("Could not save resampled wav: {}", err);
    }
}




