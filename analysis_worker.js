// Wichtig: TensorFlow.js im Worker-Kontext importieren
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js');
// NEU: Eine Bibliothek importieren, die Audio im Worker dekodieren kann
importScripts('./decode-audio-data-fast.min.js');

// Globale Konfiguration & Vorverarbeitungsfunktionen (hierher verschoben)
let model = null;

// --- Die Konfiguration und alle rechenintensiven Funktionen kommen hierher ---

const AUDIO_CONFIG = {
    SAMPLE_RATE: 16000,
    DURATION: 3,
    N_MELS: 64,
    N_FFT: 1024,
    HOP_LENGTH: 512,
    TARGET_SAMPLES: 16000 * 3,
    MIN_FREQ: 200,
    MAX_FREQ: 5000,
};

// NEU: Eine manuelle Funktion zum Resampling der Audiodaten
function resample(audioBuffer, targetSampleRate) {
    const sourceSampleRate = audioBuffer.sampleRate;
    if (sourceSampleRate === targetSampleRate) {
        return audioBuffer.getChannelData(0);
    }
    const sourceData = audioBuffer.getChannelData(0); // Mono
    const sourceLength = sourceData.length;
    const ratio = sourceSampleRate / targetSampleRate;
    const targetLength = Math.round(sourceLength / ratio);
    const resampledData = new Float32Array(targetLength);

    for (let i = 0; i < targetLength; i++) {
        const sourceIndex = i * ratio;
        const indexPrev = Math.floor(sourceIndex);
        const indexNext = Math.min(indexPrev + 1, sourceLength - 1);
        const t = sourceIndex - indexPrev;
        resampledData[i] = (1 - t) * sourceData[indexPrev] + t * sourceData[indexNext];
    }
    return resampledData;
}


// GEÄNDERT: Diese Funktion nutzt nun die neue Bibliothek und die manuelle Resampling-Funktion
async function decodeAndStandardizeAudio(file) {
    const arrayBuffer = await file.arrayBuffer();
    // decodeAudioData kommt von der importierten Bibliothek
    const audioBuffer = await self.decodeAudioData(arrayBuffer);
    
    // Resampling auf unsere Ziel-Sample-Rate
    const monoData = resample(audioBuffer, AUDIO_CONFIG.SAMPLE_RATE);
    return monoData;
}


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
                const denominator = center - leftEdge;
                if (denominator > 0) weight = (specHz - leftEdge) / denominator;
            } else if (specHz > center && specHz <= rightEdge) {
                const denominator = rightEdge - center;
                if (denominator > 0) weight = (rightEdge - specHz) / denominator;
            }
            melWeights.set(weight, j, i);
        }
    }
    return melWeights.toTensor();
}

function audioToMelspectrogram(y) {
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
        const minBin = Math.round(AUDIO_CONFIG.MIN_FREQ * AUDIO_CONFIG.N_FFT / AUDIO_CONFIG.SAMPLE_RATE);
        const maxBin = Math.round(AUDIO_CONFIG.MAX_FREQ * AUDIO_CONFIG.N_FFT / AUDIO_CONFIG.SAMPLE_RATE);
        const bandpassMaskValues = new Float32Array(freqBinCount);
        for (let i = 0; i < freqBinCount; i++) {
            bandpassMaskValues[i] = (i >= minBin && i <= maxBin) ? 1 : 0;
        }
        const bandpassMask = tf.tensor1d(bandpassMaskValues);
        const filteredStft = stft.mul(bandpassMask);
        const powerSpectrogram = tf.square(tf.abs(filteredStft));
        const melMatrix = createMelFilterbank(AUDIO_CONFIG.N_MELS, freqBinCount, AUDIO_CONFIG.SAMPLE_RATE, 0, AUDIO_CONFIG.SAMPLE_RATE / 2);
        const melSpectrogram = tf.matMul(powerSpectrogram, melMatrix);
        const logMelSpectrogram = tf.log(melSpectrogram.add(1e-6));
        return logMelSpectrogram.expandDims(-1);
    });
}

function mergeEvents(candidates) {
    if (!candidates || candidates.length === 0) return [];
    candidates.sort((a, b) => a.start - b.start);
    const merged = [Object.assign({}, candidates[0])];
    for (let i = 1; i < candidates.length; i++) {
        const last = merged[merged.length - 1];
        const current = candidates[i];
        if (current.start <= last.end) {
            last.end = Math.max(last.end, current.end);
            last.prob = Math.max(last.prob, current.prob);
        } else {
            merged.push(Object.assign({}, current));
        }
    }
    return merged;
}


// Der Haupt-Event-Listener für den Worker. Hier startet die Magie.
self.onmessage = async (event) => {
    const { files, params, modelPath } = event.data;

    try {
        if (!model) {
            postMessage({ type: 'progress', data: { text: 'Lade Erkennungsmodell...' } });
            model = await tf.loadLayersModel(modelPath);
            // "Warm-up"
            const specShape = model.inputs[0].shape.slice(1);
            const dummyInput = tf.zeros([1, ...specShape]);
            model.predict(dummyInput).dispose();
            dummyInput.dispose();
        }

        const { threshold, overlapPercent, shouldMerge } = params;
        const strideSec = AUDIO_CONFIG.DURATION * (1 - overlapPercent / 100.0);
        const strideSamples = Math.round(strideSec * AUDIO_CONFIG.SAMPLE_RATE);
        let allEvents = [];

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            postMessage({
                type: 'progress',
                data: { text: `Analysiere: ${file.name} (${i + 1}/${files.length})`, progress: (i / files.length) * 100 }
            });

            const y = await decodeAndStandardizeAudio(file);
            const specs = [];
            const starts = [];

            for (let start = 0; start + AUDIO_CONFIG.TARGET_SAMPLES <= y.length; start += strideSamples) {
                let clip = y.slice(start, start + AUDIO_CONFIG.TARGET_SAMPLES);
                const spec = audioToMelspectrogram(clip);
                specs.push(spec);
                starts.push(start);
            }

            postMessage({
                type: 'progress',
                data: { text: `Führe Erkennung für ${file.name} aus...`, progress: ((i + 0.5) / files.length) * 100 }
            });

            if (specs.length > 0) {
                const X = tf.stack(specs);
                const probs = await model.predict(X).data();
                tf.dispose(X);
                tf.dispose(specs);

                let candidates = [];
                for (let j = 0; j < probs.length; j++) {
                    if (probs[j] >= threshold) {
                        candidates.push({
                            start: starts[j],
                            end: starts[j] + AUDIO_CONFIG.TARGET_SAMPLES,
                            prob: probs[j],
                            file: file.name,
                            // WICHTIG: Sende den rohen Audioclip zurück zur Anzeige
                            audioClip: y.slice(starts[j], starts[j] + AUDIO_CONFIG.TARGET_SAMPLES)
                        });
                    }
                }
                const fileEvents = shouldMerge ? mergeEvents(candidates) : candidates;
                allEvents.push(...fileEvents);
            }
        }

        postMessage({
            type: 'progress',
            data: { text: 'Analyse abgeschlossen!', progress: 100 }
        });

        // Sende das Endergebnis an den Hauptthread
        postMessage({ type: 'done', data: allEvents });

    } catch (error) {
        console.error("Fehler im Worker:", error);
        postMessage({ type: 'error', data: error.message });
    }
};
