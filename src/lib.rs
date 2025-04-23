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

fn pad_or_truncate(mut input: Vec<f32>, target_len: usize, frame_length: usize) -> Result<Vec<f32>, String> {
    if input.len() > target_len {
        input.truncate(target_len);
    } else if input.len() < target_len {
        input.resize(target_len, 0.0);
    }

    let pad_each = frame_length / 2;
    let mut out = Vec::with_capacity(pad_each + input.len() + pad_each);
    out.extend(std::iter::repeat(0.0).take(pad_each));
    out.extend(input);
    out.extend(std::iter::repeat(0.0).take(pad_each));
    Ok(out)
}

fn normalize(input: Vec<f32>) -> Result<Vec<f32>, String> {
    let mean = input.iter().copied().sum::<f32>() / input.len() as f32;
    let variance = input.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / input.len() as f32;
    let std = variance.sqrt();

    if std == 0.0 {
        return Ok(vec![0.0; input.len()]);
    }

    Ok(input.into_iter().map(|x| (x - mean) / std).collect())
} 

fn frame_signal(input: Vec<f32>, frame_length: usize, hop_length: usize) -> Result<Vec<Vec<f32>>, String> {
    let num_frames = (input.len() - frame_length + hop_length) / hop_length;
    let mut frames = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let start = i * hop_length;
        let end = start + frame_length;

        if end <= input.len() {
            frames.push(input[start..end].to_vec());
        }
    }

    Ok(frames)
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

    let padded = match pad_or_truncate(resampled, 480000, 400) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{}", err);
            return;
        }
    };

    let normalized = match normalize(padded) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{}", err);
            return;
        }
    };

    println!("Final sample count: {}", normalized.len());

    if let Err(err) = save_wav("resampled.wav", &normalized, 16000) {
        eprintln!("Could not save resampled wav: {}", err);
    }

    let framed = match frame_signal(normalized, 400, 160) {
        Ok(v) => v,
        Err(err) => {
            eprint!("{}", err);
            return;
        }
    };

    eprint!("Framed audio has {} frames of size {}", framed.len(), framed[0].len());

}




