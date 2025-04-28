use hound::{WavReader, WavSpec, SampleFormat, WavWriter};
use std::{ffi::CStr, i16};
use std::os::raw::c_char;
use rubato::{SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction, Resampler};
use realfft::RealFftPlanner;
use num_complex::Complex;

fn type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<&T>()
}

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

fn resample_audio(samples: Vec<f32>, orig_rate: u32, target_rate: u32) -> Result<Vec<f32>, String> {
    if orig_rate == target_rate {
        return Ok(samples)
    }

    let ratio = target_rate as f64 / orig_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, samples.len(), 1)
        .map_err(|e| format!("Resampler init error: {:?}", e))?;
    let outputs = resampler
        .process(&[samples], None)
        .map_err(|e| format!("Resampler init error: {:?}", e))?;
    Ok(outputs.into_iter().next().unwrap())
}

fn pad_or_truncate(mut samples: Vec<f32>, target_len: usize, frame_length: usize) -> Result<Vec<f32>, String> {
    if samples.len() > target_len {
        samples.truncate(target_len);
    } else if samples.len() < target_len {
        samples.resize(target_len, 0.0);
    }

    let pad_each = frame_length / 2;
    let mut out = Vec::with_capacity(pad_each + samples.len() + pad_each);
    out.extend(std::iter::repeat(0.0).take(pad_each));
    out.extend(samples);
    out.extend(std::iter::repeat(0.0).take(pad_each));
    Ok(out)
}

fn normalize(samples: Vec<f32>) -> Result<Vec<f32>, String> {
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    let variance = samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / samples.len() as f32;
    let std = variance.sqrt();

    if std == 0.0 {
        return Ok(vec![0.0; samples.len()]);
    }

    Ok(samples.into_iter().map(|x| (x - mean) / std).collect())
} 

fn frame_signal(samples: Vec<f32>, frame_length: usize, hop_length: usize) -> Result<Vec<Vec<f32>>, String> {
    let num_frames = (samples.len() - frame_length + hop_length) / hop_length;
    let mut frames = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let start = i * hop_length;
        let end = start + frame_length;

        if end <= samples.len() {
            frames.push(samples[start..end].to_vec());
        }
    }

    Ok(frames)
}

fn apply_hann_window(mut frames: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>, String> {
    let frame_len = frames[0].len();
    let hann: Vec<f32> = (0..frame_len)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / frame_len as f32).cos())
        .collect();
    
    for frame in frames.iter_mut() {
        for (i, sample) in frame.iter_mut().enumerate() {
            *sample *= hann[i]
        }
    }

    Ok(frames)
}

fn apply_rfft(frames: Vec<Vec<f32>>) -> Result<Vec<Vec<Complex<f32>>>, String> {
    let frame_len = frames[0].len();
    
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(frame_len);

    let mut output = r2c.make_output_vec();
    let mut spectrogram = Vec::with_capacity(frames.len());

    for frame in frames {

        let mut input = r2c.make_input_vec();
        input.copy_from_slice(&frame);

        r2c.process(&mut input, &mut output)
            .map_err(|e| format!("FFT error: {:?}", e))?;

        spectrogram.push(output.clone());
    }

    Ok(spectrogram)
}

fn power_spectrogram(spectrogram: Vec<Vec<Complex<f32>>>) -> Result<Vec<Vec<f32>>, String> {
    Ok(spectrogram.into_iter()
        .map(|frame|
            frame.into_iter()
                .map(|c| c.norm_sqr())
                .collect()
        ).collect()
    )
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

fn hertz_to_mel_htk(frequency: f32) -> f32 {
    2595.0 * (1.0 + frequency / 700.0).log10()
}

fn mel_to_hertz_htk(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn linspace(start: f32, end: f32, num: usize) -> Vec<f32> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (end - start) / (num - 1) as f32;
    (0..num).map(|i| start + step * i as f32).collect()
}

fn create_triangular_filter_bank(fft_freqs: &[f32], filter_freqs: &[f32]) -> Result<Vec<Vec<f32>>, String> {
    let num_frequency_bins = fft_freqs.len();
    let num_mel_filters = filter_freqs.len() - 2;

    if num_mel_filters <= 0 {
        return Err("Number of mel filters must be positive.".to_string());
    }

    let filter_diffs: Vec<f32> = filter_freqs.windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();

    let mut mel_filters = vec![vec![0.0f32; num_mel_filters]; num_frequency_bins];

    for (k, &fft_freq) in fft_freqs.iter().enumerate() {
        let mut slopes = vec![0.0f32; filter_freqs.len()];
        for (i, &filter_freq) in filter_freqs.iter().enumerate() {
            slopes[i] = filter_freq - fft_freq;
        }

        for m in 0..num_mel_filters {
            let down_slope = if filter_diffs[m] != 0.0 {
                -slopes[m] / filter_diffs[m]
            } else {
                0.0
            };
            let up_slope = if filter_diffs[m + 1] != 0.0 {
                slopes[m + 2] / filter_diffs[m + 1]
            } else {
                0.0
            };
            mel_filters[k][m] = down_slope.min(up_slope).max(0.0);
        }
    }

    Ok(mel_filters)
}

fn mel_filter_bank(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    min_frequency: f32,
    max_frequency: f32,
    sampling_rate: u32,
) -> Result<Vec<Vec<f32>>, String> {
    let nyquist = sampling_rate as f32 / 2.0;
    let mel_min = hertz_to_mel_htk(min_frequency);
    let mel_max = hertz_to_mel_htk(max_frequency);

    let mel_freqs_vec = linspace(mel_min, mel_max, num_mel_filters + 2);

    let mut filter_freqs_hz: Vec<f32> = Vec::with_capacity(mel_freqs_vec.len());

    for mel in mel_freqs_vec.iter() {
        filter_freqs_hz.push(mel_to_hertz_htk(*mel));
    }

    let fft_freqs: Vec<f32>;
    let filter_freqs_for_triangulation: &[f32];
//    let fft_bin_width = sampling_rate as f32 / ((num_frequency_bins - 1) as f32 * 2.0);
    let fft_freqs_hz = linspace(0.0, nyquist, num_frequency_bins);

    let mut fft_freqs_mel = Vec::with_capacity(num_frequency_bins);
    for freq_hz in fft_freqs_hz.iter() {
        fft_freqs_mel.push(hertz_to_mel_htk(*freq_hz));
    }
    fft_freqs = fft_freqs_mel;
    filter_freqs_for_triangulation = &mel_freqs_vec;

    let mel_filters = create_triangular_filter_bank(&fft_freqs, filter_freqs_for_triangulation)?;

    Ok(mel_filters)
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

    eprintln!("Framed audio has {} frames of size {}", framed.len(), framed[0].len());

    let hann_weighted = match apply_hann_window(framed) {
        Ok(v) => v,
        Err(err) => {
            eprint!("{}", err);
            return;
        }
    };

    let rfft_spectrogram = match apply_rfft(hann_weighted) {
        Ok(v) => v,
        Err(err) => {
            eprint!("{}", err);
            return;
        }
    };

    eprintln!("RFFT spectrogram type: {}", type_of(&rfft_spectrogram));
    eprintln!("RFFT Spectrogram shape: ({} x {})", rfft_spectrogram.len(), rfft_spectrogram[0].len());

    let power_spec = match power_spectrogram(rfft_spectrogram) {
        Ok(v) => v,
        Err(err) => {
            eprint!("{}", err);
            return;
        }
    };

    eprintln!("Power spectrogram type: {}", type_of(&power_spec));
    eprintln!("Power spectrogram shape: ({} x {})", power_spec.len(), power_spec[0].len());

    let mel_filters = match mel_filter_bank(201, 80, 0.0, 8000.0, 16000) {
        Ok(v) => v,
        Err(err) => {
            eprint!("{}", err);
            return;
        }
    };

    println!("Mel filter bank shape: ({} x {})", mel_filters.len(), mel_filters[0].len());
    
    let mut mel_spectrogram = vec![vec![0.0f32; 80]; 3001];

    for i in 0..3001 {
        for m in 0..80 {
            let mut sum = 0.0;
            for k in 0..201 {
                sum += power_spec[i][k] * mel_filters[k][m];
            }
        mel_spectrogram[i][m] = sum;
        }
    }

    println!("Mel spectrogram shape: ({} x {})", mel_spectrogram.len(), mel_spectrogram[0].len());
}




