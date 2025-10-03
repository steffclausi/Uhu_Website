export const ESTIMATED_PROCESSING_RATE_MIN_PER_HOUR = 1788; 
export const MODEL_PATH = './uhu_model_js/model.json';
export const MAX_CLIPS_TO_SHOW_PER_CATEGORY = 20;
export const UI_BUFFER_MS = 120; // Puffer in Millisekunden, um die UI flüssig zu halten
export const AUDIO_CONFIG = {
    SAMPLE_RATE: 16000,
    DURATION: 3,
    N_MELS: 64,
    N_FFT: 1024,
    HOP_LENGTH: 512,
    TARGET_SAMPLES: 16000 * 3,
    MIN_FREQ: 200,
    MAX_FREQ: 5000,
};
export const DETECTION_THRESHOLD = 0.95;	