import { AUDIO_CONFIG } from './config.js';

export async function decodeAndStandardizeAudio(file) {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: AUDIO_CONFIG.SAMPLE_RATE });
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
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

function createMelFilterbank(numMelBins, numSpectrogramBins, sampleRate) {
    const hzToMel = (hz) => 1127.0 * Math.log(1.0 + hz / 700.0);
    const melToHz = (mel) => 700.0 * (Math.exp(mel / 1127.0) - 1.0);
    const lowerMel = hzToMel(0);
    const upperMel = hzToMel(sampleRate / 2.0);
    const melPoints = tf.linspace(lowerMel, upperMel, numMelBins + 2).arraySync();
    const hzPoints = melPoints.map(melToHz);
    const spectrogramBinHz = sampleRate / 2.0 / (numSpectrogramBins - 1);
    const spectrogramBinEdges = Array.from({ length: numSpectrogramBins }, (_, i) => i * spectrogramBinHz);
    const melWeights = tf.buffer([numSpectrogramBins, numMelBins]);
    for (let i = 0; i < numMelBins; i++) {
        const leftEdge = hzPoints[i], center = hzPoints[i + 1], rightEdge = hzPoints[i + 2];
        for (let j = 0; j < numSpectrogramBins; j++) {
            const specHz = spectrogramBinEdges[j];
            let weight = 0.0;
            if (specHz >= leftEdge && specHz <= center) weight = (specHz - leftEdge) / (center - leftEdge);
            else if (specHz > center && specHz <= rightEdge) weight = (rightEdge - specHz) / (rightEdge - center);
            if (weight > 0) melWeights.set(weight, j, i);
        }
    }
    return melWeights.toTensor();
}

export function audioToMelspectrogram(y) {
    const yPadded = new Float32Array(AUDIO_CONFIG.TARGET_SAMPLES).fill(0);
    const toCopy = Math.min(y.length, AUDIO_CONFIG.TARGET_SAMPLES);
    yPadded.set(y.slice(0, toCopy));

    return tf.tidy(() => {
        const stft = tf.signal.stft(tf.tensor1d(yPadded), AUDIO_CONFIG.N_FFT, AUDIO_CONFIG.HOP_LENGTH);
        const freqBinCount = AUDIO_CONFIG.N_FFT / 2 + 1;
        const minBin = Math.round(AUDIO_CONFIG.MIN_FREQ * AUDIO_CONFIG.N_FFT / AUDIO_CONFIG.SAMPLE_RATE);
        const maxBin = Math.round(AUDIO_CONFIG.MAX_FREQ * AUDIO_CONFIG.N_FFT / AUDIO_CONFIG.SAMPLE_RATE);
        const bandpassMask = tf.tensor1d(new Float32Array(freqBinCount).map((_, i) => (i >= minBin && i <= maxBin) ? 1 : 0));
        const powerSpectrogram = tf.square(tf.abs(stft.mul(bandpassMask)));
        const melMatrix = createMelFilterbank(AUDIO_CONFIG.N_MELS, freqBinCount, AUDIO_CONFIG.SAMPLE_RATE);
        const melSpectrogram = tf.matMul(powerSpectrogram, melMatrix);
        return tf.log(melSpectrogram.add(1e-6)).expandDims(-1);
    });
}