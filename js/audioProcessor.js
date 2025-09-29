import { AUDIO_CONFIG } from './config.js';

/**
 * Dekodiert eine Audiodatei und standardisiert sie (Mono, definierte Sample-Rate).
 * @param {File} file - Die Audiodatei.
 * @returns {Promise<Float32Array>} Die Audiodaten.
 */
export async function decodeAndStandardizeAudio(file) {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: AUDIO_CONFIG.SAMPLE_RATE });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    // In Mono umwandeln, falls nötig
    if (audioBuffer.numberOfChannels > 1) {
        const offlineCtx = new OfflineAudioContext(1, audioBuffer.duration * AUDIO_CONFIG.SAMPLE_RATE, AUDIO_CONFIG.SAMPLE_RATE);
        const source = offlineCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineCtx.destination);
        source.start(0);
        const renderedBuffer = await offlineCtx.startRendering();
        return renderedBuffer.getChannelData(0);
    }

    return audioBuffer.getChannelData(0);
}

/**
 * Erstellt eine Mel-Filterbank-Matrix.
 */
function createMelFilterbank(numMelBins, numSpectrogramBins, sampleRate, lowerEdgeHz, upperEdgeHz) {
    const hzToMel = (hz) => 1127.0 * Math.log(1.0 + hz / 700.0);
    const melToHz = (mel) => 700.0 * (Math.exp(mel / 1127.0) - 1.0);
    const lowerMel = hzToMel(lowerEdgeHz);
    const upperMel = hzToMel(upperEdgeHz);
    const melPoints = tf.linspace(lowerMel, upperMel, numMelBins + 2).arraySync();
    const hzPoints = melPoints.map(melToHz);
    const spectrogramBinHz = sampleRate / 2.0 / (numSpectrogramBins - 1);
    const spectrogramBinEdges = Array.from({ length: numSpectrogramBins }, (_, i) => i * spectrogramBinHz);
    const melWeights = tf.buffer([numSpectrogramBins, numMelBins]);

    for (let i = 0; i < numMelBins; i++) {
        const leftEdge = hzPoints[i];
        const center = hzPoints[i + 1];
        const rightEdge = hzPoints[i + 2];
        for (let j = 0; j < numSpectrogramBins; j++) {
            const specHz = spectrogramBinEdges[j];
            let weight = 0.0;
            if (specHz >= leftEdge && specHz <= center) {
                weight = (specHz - leftEdge) / (center - leftEdge);
            } else if (specHz > center && specHz <= rightEdge) {
                weight = (rightEdge - specHz) / (rightEdge - center);
            }
            melWeights.set(weight, j, i);
        }
    }
    return melWeights.toTensor();
}

/**
 * Konvertiert einen rohen Audio-Array in ein Mel-Spektrogramm-Tensor.
 * @param {Float32Array} y - Der Audio-Array.
 * @returns {tf.Tensor} Der resultierende Tensor.
 */
export function audioToMelspectrogram(y) {
    if (y.length > AUDIO_CONFIG.TARGET_SAMPLES) {
        y = y.slice(0, AUDIO_CONFIG.TARGET_SAMPLES);
    } else if (y.length < AUDIO_CONFIG.TARGET_SAMPLES) {
        const padding = new Float32Array(AUDIO_CONFIG.TARGET_SAMPLES - y.length).fill(0);
        const newY = new Float32Array(AUDIO_CONFIG.TARGET_SAMPLES);
        newY.set(y);
        newY.set(padding, y.length);
        y = newY;
    }

    return tf.tidy(() => {
        const inputTensor = tf.tensor1d(y);
        const stft = tf.signal.stft(inputTensor, AUDIO_CONFIG.N_FFT, AUDIO_CONFIG.HOP_LENGTH);
        const freqBinCount = AUDIO_CONFIG.N_FFT / 2 + 1;
        
        // Bandpass-Filter anwenden
        const minBin = Math.round(AUDIO_CONFIG.MIN_FREQ * AUDIO_CONFIG.N_FFT / AUDIO_CONFIG.SAMPLE_RATE);
        const maxBin = Math.round(AUDIO_CONFIG.MAX_FREQ * AUDIO_CONFIG.N_FFT / AUDIO_CONFIG.SAMPLE_RATE);
        const bandpassMaskValues = new Float32Array(freqBinCount).map((_, i) => (i >= minBin && i <= maxBin) ? 1 : 0);
        const bandpassMask = tf.tensor1d(bandpassMaskValues);
        
        const filteredStft = stft.mul(bandpassMask);
        
        const powerSpectrogram = tf.square(tf.abs(filteredStft));
        
        const melMatrix = createMelFilterbank(
            AUDIO_CONFIG.N_MELS, freqBinCount, AUDIO_CONFIG.SAMPLE_RATE, 0, AUDIO_CONFIG.SAMPLE_RATE / 2
        );
        
        const melSpectrogram = tf.matMul(powerSpectrogram, melMatrix);
        const logMelSpectrogram = tf.log(melSpectrogram.add(1e-6));
        
        return logMelSpectrogram.expandDims(-1);
    });
}