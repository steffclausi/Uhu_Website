import { MODEL_PATH, AUDIO_CONFIG, DETECTION_THRESHOLD } from './config.js';
import { UIElements, handleFiles, displayResults } from './ui.js';
import { decodeAndStandardizeAudio, audioToMelspectrogram } from './audioProcessor.js';
import { formatRemainingTime, mergeEvents } from './utils.js';

// --- Globaler Zustand ---
let uploadedFiles = [];
let fileDurations = [];
let model = null;

// --- Event Listeners ---
// Sicherstellen, dass die Elemente existieren, bevor Listener angehängt werden
if (UIElements.fileUploader) {
    UIElements.fileUploader.addEventListener('change', async (e) => {
        const fileData = await handleFiles(e.target.files);
        uploadedFiles = fileData.uploadedFiles;
        fileDurations = fileData.fileDurations;
    });
}

if (UIElements.dropZone) {
    UIElements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        UIElements.dropZone.classList.add('border-indigo-500', 'bg-indigo-50');
    });

    UIElements.dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        UIElements.dropZone.classList.remove('border-indigo-500', 'bg-indigo-50');
    });

    UIElements.dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        UIElements.dropZone.classList.remove('border-indigo-500', 'bg-indigo-50');
        const fileData = await handleFiles(e.dataTransfer.files);
        uploadedFiles = fileData.uploadedFiles;
        fileDurations = fileData.fileDurations;
    });
}

UIElements.overlapSlider.addEventListener('input', (e) => {
    UIElements.overlapValue.textContent = `${e.target.value}%`;
});

UIElements.startBtn.addEventListener('click', runAnalysis);


// --- Hauptfunktionen ---
async function loadModel() {
    if (model) return model;
    
    UIElements.progressText.textContent = 'Lade Erkennungsmodell...';
    try {
        model = await tf.loadLayersModel(MODEL_PATH);
        const specShape = model.inputs[0].shape.slice(1);
        const dummyInput = tf.zeros([1, ...specShape]);
        model.predict(dummyInput).dispose();
        dummyInput.dispose();
        return model;
    } catch (error) {
        UIElements.progressText.textContent = `Fehler beim Laden des Modells: ${MODEL_PATH} nicht gefunden.`;
        console.error("Modell-Ladefehler:", error);
        throw error;
    }
}

async function runAnalysis() {
    if (uploadedFiles.length === 0) {
        alert("Bitte wählen Sie zuerst Audiodateien aus.");
        return;
    }

    UIElements.startBtn.disabled = true;
    UIElements.startBtn.classList.add('opacity-50', 'cursor-not-allowed');
    UIElements.analysisProgress.classList.remove('hidden');
    UIElements.resultsSection.classList.add('hidden');
    UIElements.resultsGrid.innerHTML = '';
    UIElements.resultsSummary.innerHTML = '';
    UIElements.timeStats.innerHTML = '';

    try {
        await loadModel();

        const totalAudioDurationSec = fileDurations.reduce((sum, d) => sum + d, 0);
        let processedAudioDurationSec = 0;
        let actualProcessingSpeedMinPerHour = null;

        const overlapPercent = parseInt(UIElements.overlapSlider.value);
        const shouldMerge = UIElements.mergeCheckbox.checked;
        const strideSec = AUDIO_CONFIG.DURATION * (1 - overlapPercent / 100.0);
        const strideSamples = Math.round(strideSec * AUDIO_CONFIG.SAMPLE_RATE);

        let allEvents = [];

        for (let i = 0; i < uploadedFiles.length; i++) {
            const file = uploadedFiles[i];
            const fileDuration = fileDurations[i];
            const fileProcessingStartTime = Date.now();
            
            UIElements.progressText.textContent = `Analysiere: ${file.name} (${i + 1}/${uploadedFiles.length})`;
            UIElements.progressBar.style.width = `${((i) / uploadedFiles.length) * 100}%`;
            
            try {
                const y = await decodeAndStandardizeAudio(file);
                const specs = [];
                const starts = [];
                for (let start = 0; start + AUDIO_CONFIG.TARGET_SAMPLES <= y.length; start += strideSamples) {
                    specs.push(audioToMelspectrogram(y.slice(start, start + AUDIO_CONFIG.TARGET_SAMPLES)));
                    starts.push(start);
                }

                if (specs.length > 0) {
                    const X = tf.stack(specs);
                    const probs = await model.predict(X).data();
                    tf.dispose(X);
                    tf.dispose(specs);

                    let candidates = [];
                    for (let j = 0; j < probs.length; j++) {
                        if (probs[j] >= DETECTION_THRESHOLD) {
                            candidates.push({
                                start: starts[j],
                                end: starts[j] + AUDIO_CONFIG.TARGET_SAMPLES,
                                prob: probs[j],
                                file: file.name,
                                audioClip: y.slice(starts[j], starts[j] + AUDIO_CONFIG.TARGET_SAMPLES)
                            });
                        }
                    }
                    allEvents.push(...(shouldMerge ? mergeEvents(candidates) : candidates));
                }
                const fileProcessingTimeSec = (Date.now() - fileProcessingStartTime) / 1000;
                processedAudioDurationSec += fileDuration;
                if (i === 0 && fileProcessingTimeSec > 0) {
                    actualProcessingSpeedMinPerHour = (fileDuration / fileProcessingTimeSec) * 60;
                }
                if (actualProcessingSpeedMinPerHour) {
                    const remainingAudioSec = totalAudioDurationSec - processedAudioDurationSec;
                    const speedSecPerSec = actualProcessingSpeedMinPerHour / 60;
                    const remainingTimeSec = speedSecPerSec > 0 ? remainingAudioSec / speedSecPerSec : Infinity;
                    UIElements.timeStats.innerHTML = `${formatRemainingTime(remainingTimeSec)}<span class="mx-2 text-gray-400">|</span>Geschwindigkeit: ${actualProcessingSpeedMinPerHour.toFixed(0)} min Audio/Std.`;
                }
            } catch (e) {
                console.error(`Fehler bei der Verarbeitung von ${file.name}:`, e);
                alert(`Konnte Datei '${file.name}' nicht verarbeiten: ${e.message}`);
            }
        }
        
        UIElements.progressBar.style.width = '100%';
        UIElements.progressText.textContent = 'Analyse abgeschlossen!';
        displayResults(allEvents);

    } catch (error) {
        console.error("Ein Fehler ist aufgetreten:", error);
        UIElements.progressText.textContent = `Ein Fehler ist aufgetreten: ${error.message}`;
    } finally {
        UIElements.startBtn.disabled = false;
        UIElements.startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        setTimeout(() => {
            UIElements.analysisProgress.classList.add('hidden');
            UIElements.progressBar.style.width = '0%';
        }, 3000);
    }
}