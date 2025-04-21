use hound::WavReader;
use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn read_wav_info(path: *const c_char) {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Invalid path string");
            return;
        }
    };

    let reader = match WavReader::open(path_str) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to read wav file: {}", e);
            return
        }
    };

    let spec = reader.spec();
    println!("Sampling rate:  {}", spec.sample_rate);
    println!("Bits per sampmle: {}", spec.bits_per_sample);
    println!("Num channels: {}", spec.channels);

    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();

    println!("Loaded {} samples", samples.len());
}
