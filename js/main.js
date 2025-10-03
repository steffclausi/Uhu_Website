import { MODEL_PATH, AUDIO_CONFIG } from './config.js';
import { UIElements, handleFiles, displayResults } from './ui.js';
import { decodeAndStandardizeAudio, audioToMelspectrogram } from './audioProcessor.js';
import { mergeEvents, formatRemainingTime } from './utils.js';

let model;
let uploadedFiles = [];
let fileDurations = [];
let isStopRequested = false;

async function loadModel() {
    if (model) return;
    console.log("Loading model...");
    UIElements.progressText.textContent = 'Initialisiere und lade KI-Modell...';
    try {
        model = await tf.loadLayersModel(MODEL_PATH);
        console.log("Model loaded successfully.");
        UIElements.progressText.textContent = 'Modell initialisiert.';
    } catch (error) {
        console.error("Error loading model:", error);
        UIElements.progressText.textContent = `Fehler beim Laden des Modells: ${error.message}`;
        throw error;
    }
}

async function runAnalysis() {
    if (uploadedFiles.length === 0) {
        alert("Bitte wählen Sie zuerst Audiodateien aus.");
        return;
    }

    isStopRequested = false;
    UIElements.startBtn.disabled = true;
    UIElements.startBtn.classList.add('hidden');
    UIElements.stopBtn.classList.remove('hidden');
    UIElements.stopBtn.disabled = false;
    UIElements.stopBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    UIElements.analysisProgress.classList.remove('hidden');
    UIElements.resultsSection.classList.add('hidden');
    UIElements.resultsGrid.innerHTML = '';
    UIElements.resultsSummary.innerHTML = '';
    UIElements.timeStats.innerHTML = '';

    const uiBufferMs = parseInt(UIElements.bufferSlider.value);
    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    try {
        console.log("Analysis started.");
        await loadModel();
        console.log("Model loaded.");

        const overlapPercent = parseInt(UIElements.overlapSlider.value);
        const detectionThreshold = parseFloat(UIElements.confidenceSlider.value);
        const shouldMerge = UIElements.mergeCheckbox.checked;
        const strideSec = AUDIO_CONFIG.DURATION * (1 - overlapPercent / 100.0);
        const strideSamples = Math.round(strideSec * AUDIO_CONFIG.SAMPLE_RATE);
        const BATCH_SIZE = 24;

        let totalClipsToProcess = 0;
        for (let i = 0; i < fileDurations.length; i++) {
            const fileDuration = fileDurations[i];
            const numClipsInFile = Math.floor((fileDuration - AUDIO_CONFIG.DURATION) / strideSec) + 1;
            if (numClipsInFile > 0) {
                totalClipsToProcess += numClipsInFile;
            }
        }
        console.log(`Total clips to process: ${totalClipsToProcess}`);
        
        let allEvents = [];
        let processedClips = 0;
        const analysisStartTime = Date.now();

        analysisLoop: for (let i = 0; i < uploadedFiles.length; i++) {
            const file = uploadedFiles[i];
            UIElements.progressText.textContent = `Analysiere: ${file.name} (${i + 1}/${uploadedFiles.length})`;
            console.log(`Processing file ${i + 1}/${uploadedFiles.length}: ${file.name}`);

            try {
                const y = await decodeAndStandardizeAudio(file);
                console.log(`File ${file.name} decoded.`);
                await sleep(uiBufferMs);

                let specs = [];
                let startsInSamples = [];
                let fileCandidates = [];

                for (let start = 0; start + AUDIO_CONFIG.TARGET_SAMPLES <= y.length; start += strideSamples) {
                    if (isStopRequested) {
                        console.log("Stop requested, finishing current batch...");
                        break analysisLoop;
                    }

                    specs.push(audioToMelspectrogram(y.slice(start, start + AUDIO_CONFIG.TARGET_SAMPLES)));
                    startsInSamples.push(start);

                    const isLastClipInFile = start + strideSamples + AUDIO_CONFIG.TARGET_SAMPLES > y.length;
                    if (specs.length === BATCH_SIZE || (isLastClipInFile && specs.length > 0)) {
                        const batchSize = specs.length;
                        console.log(`Processing batch of ${batchSize} clips...`);

                        const X = tf.stack(specs);
                        const probs = await model.predict(X).data();
                        tf.dispose(X);
                        tf.dispose(specs);
                        specs = [];
                        console.log("Batch prediction complete.");

                        for (let j = 0; j < probs.length; j++) {
                            if (probs[j] >= detectionThreshold) {
                                const clipStart = startsInSamples[j];
                                const clipEnd = clipStart + AUDIO_CONFIG.TARGET_SAMPLES;
                                fileCandidates.push({
                                    start: clipStart,
                                    end: clipEnd,
                                    prob: probs[j],
                                    file: file.name,
                                    audioClip: y.slice(clipStart, clipEnd)
                                });
                            }
                        }
                        startsInSamples = [];

                        processedClips += batchSize;

                        const progress = totalClipsToProcess > 0 ? processedClips / totalClipsToProcess : 0;
                        UIElements.progressBar.style.width = `${progress * 100}%`;

                        const elapsedTimeSec = (Date.now() - analysisStartTime) / 1000;
                        if (elapsedTimeSec > 1) {
                            const remainingClips = totalClipsToProcess - processedClips;
                            const clipsPerSecond = processedClips / elapsedTimeSec;
                            const remainingTimeSec = clipsPerSecond > 0 ? remainingClips / clipsPerSecond : Infinity;
                            
                            const processedAudioDurationSec = processedClips * AUDIO_CONFIG.DURATION;
                            const speedHrPerMin = elapsedTimeSec > 0 ? (processedAudioDurationSec / 3600) / (elapsedTimeSec / 60) : 0;

                            UIElements.timeStats.innerHTML = `${formatRemainingTime(remainingTimeSec)}<span class="mx-2 text-gray-400">|</span>Geschwindigkeit: ${speedHrPerMin.toFixed(2)} Std. Audio/Min.`;
                        }

                        await sleep(uiBufferMs);
                    }
                }
                allEvents.push(...(shouldMerge ? mergeEvents(fileCandidates) : fileCandidates));

            } catch (e) {
                console.error(`Fehler bei der Verarbeitung von ${file.name}:`, e);
                alert(`Konnte Datei '''${file.name}''' nicht verarbeiten: ${e.message}`);
            }
        }
        
        if (isStopRequested) {
            UIElements.progressText.textContent = 'Analyse gestoppt.';
        } else {
            UIElements.progressBar.style.width = '100%';
            UIElements.progressText.textContent = 'Analyse abgeschlossen!';
        }
        console.log("Analysis finished or stopped.");

        const finalElapsedTimeSec = (Date.now() - analysisStartTime) / 1000;
        if (finalElapsedTimeSec > 1 && processedClips > 0) { // Use processedClips instead of totalClipsToProcess
            const finalProcessedAudioDurationSec = processedClips * AUDIO_CONFIG.DURATION;
            const finalSpeedHrPerMin = (finalProcessedAudioDurationSec / 3600) / (finalElapsedTimeSec / 60);
            localStorage.setItem('uhuAnalysisSpeedHrPerMin', finalSpeedHrPerMin);
            console.log(`Saved analysis speed to localStorage: ${finalSpeedHrPerMin}`);
        }

        displayResults(allEvents, detectionThreshold);

    } catch (error) {
        console.error("Ein Fehler ist aufgetreten:", error);
        UIElements.progressText.textContent = `Ein Fehler ist aufgetreten: ${error.message}`;
    } finally {
        UIElements.startBtn.disabled = false;
        UIElements.startBtn.classList.remove('hidden', 'opacity-50', 'cursor-not-allowed');
        UIElements.stopBtn.classList.add('hidden');
        setTimeout(() => {
            UIElements.analysisProgress.classList.add('hidden');
            UIElements.progressBar.style.width = '0%';
            UIElements.timeStats.innerHTML = '';
        }, 3000);
    }
}

// --- Event Listeners ---
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
        UIElements.dropZone.classList.add('border-indigo-600');
    });

    UIElements.dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        UIElements.dropZone.classList.remove('border-indigo-600');
    });

    UIElements.dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        UIElements.dropZone.classList.remove('border-indigo-600');
        const fileData = await handleFiles(e.dataTransfer.files);
        uploadedFiles = fileData.uploadedFiles;
        fileDurations = fileData.fileDurations;
    });
}

if (UIElements.startBtn) {
    UIElements.startBtn.addEventListener('click', runAnalysis);
}

if (UIElements.stopBtn) {
    UIElements.stopBtn.addEventListener('click', () => {
        isStopRequested = true;
        UIElements.progressText.textContent = 'Analyse wird nach dem aktuellen Batch gestoppt...';
        UIElements.stopBtn.disabled = true;
        UIElements.stopBtn.classList.add('opacity-50', 'cursor-not-allowed');
    });
}

if (UIElements.overlapSlider) {
    UIElements.overlapSlider.addEventListener('input', (e) => {
        if (UIElements.overlapValue) {
            UIElements.overlapValue.textContent = `${e.target.value}%`;
        }
    });
}

if (UIElements.confidenceSlider) {
    UIElements.confidenceSlider.addEventListener('input', (e) => {
        if (UIElements.confidenceValue) {
            UIElements.confidenceValue.textContent = `${Math.round(parseFloat(e.target.value) * 100)}%`;
        }
    });
}

if (UIElements.bufferSlider) {
    UIElements.bufferSlider.addEventListener('input', (e) => {
        if (UIElements.bufferValue) {
            UIElements.bufferValue.textContent = `${e.target.value}ms`;
        }
    });
}
