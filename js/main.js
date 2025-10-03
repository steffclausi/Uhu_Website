import { MODEL_PATH, AUDIO_CONFIG, DETECTION_THRESHOLD, UI_BUFFER_MS } from './config.js';

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

    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    try {
        console.log("Analysis started.");
        await loadModel();
        console.log("Model loaded.");

        const overlapPercent = parseInt(UIElements.overlapSlider.value);
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

        for (let i = 0; i < uploadedFiles.length; i++) {
            const file = uploadedFiles[i];
            UIElements.progressText.textContent = `Analysiere: ${file.name} (${i + 1}/${uploadedFiles.length})`;
            console.log(`Processing file ${i + 1}/${uploadedFiles.length}: ${file.name}`);

            try {
                const y = await decodeAndStandardizeAudio(file);
                console.log(`File ${file.name} decoded.`);
                await sleep(UI_BUFFER_MS); // Pause after decoding

                let specs = [];
                let startsInSamples = [];
                let fileCandidates = [];

                for (let start = 0; start + AUDIO_CONFIG.TARGET_SAMPLES <= y.length; start += strideSamples) {
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
                            if (probs[j] >= DETECTION_THRESHOLD) {
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

                        // UI-Updates nach jedem Batch
                        const progress = totalClipsToProcess > 0 ? processedClips / totalClipsToProcess : 0;
                        UIElements.progressBar.style.width = `${progress * 100}%`;

                        const elapsedTimeSec = (Date.now() - analysisStartTime) / 1000;
                        if (elapsedTimeSec > 1) {
                            const clipsPerSecond = processedClips / elapsedTimeSec;
                            const remainingClips = totalClipsToProcess - processedClips;
                            const remainingTimeSec = clipsPerSecond > 0 ? remainingClips / clipsPerSecond : Infinity;
                            
                            const processedAudioDurationSec = processedClips * AUDIO_CONFIG.DURATION;
                            const speedMinPerHour = elapsedTimeSec > 0 ? (processedAudioDurationSec / 60) / (elapsedTimeSec / 3600) : 0;

                            UIElements.timeStats.innerHTML = `${formatRemainingTime(remainingTimeSec)}<span class="mx-2 text-gray-400">|</span>Geschwindigkeit: ${speedMinPerHour.toFixed(0)} min Audio/Std.`;
                        }

                        await sleep(UI_BUFFER_MS); // Kurze Pause, um UI-Freezes zu verhindern
                    }
                }
                allEvents.push(...(shouldMerge ? mergeEvents(fileCandidates) : fileCandidates));

            } catch (e) {
                console.error(`Fehler bei der Verarbeitung von ${file.name}:`, e);
                alert(`Konnte Datei '${file.name}' nicht verarbeiten: ${e.message}`);
            }
        }
        
        UIElements.progressBar.style.width = '100%';
        UIElements.progressText.textContent = 'Analyse abgeschlossen!';
        console.log("Analysis finished.");
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
            UIElements.timeStats.innerHTML = '';
        }, 3000);
    }
}