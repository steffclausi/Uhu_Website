// --- Globale Konfiguration ---

// Schätzwert für die Verarbeitungsgeschwindigkeit (Minuten Audio pro Stunde Echtzeit)
export const ESTIMATED_PROCESSING_RATE_MIN_PER_HOUR = 1788; 

// Pfad zum TensorFlow.js-Modell
export const MODEL_PATH = './uhu_model_js/model.json';

// Maximale Anzahl von Clips, die pro Kategorie (hohe/mittlere Konfidenz) angezeigt werden
export const MAX_CLIPS_TO_SHOW_PER_CATEGORY = 20;

// Konfiguration für die Audioverarbeitung
export const AUDIO_CONFIG = {
    SAMPLE_RATE: 16000,
    DURATION: 3, // in Sekunden
    N_MELS: 64,
    N_FFT: 1024,
    HOP_LENGTH: 512,
    TARGET_SAMPLES: 16000 * 3,
    MIN_FREQ: 200,
    MAX_FREQ: 5000,
};

// Schwellenwert für die Erkennung
export const DETECTION_THRESHOLD = 0.5;