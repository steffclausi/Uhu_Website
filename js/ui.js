import { createWavBlob, formatTime } from './utils.js';
import { audioToMelspectrogram } from './audioProcessor.js';
import { AUDIO_CONFIG, MAX_CLIPS_TO_SHOW_PER_CATEGORY, ESTIMATED_PROCESSING_RATE_MIN_PER_HOUR } from './config.js';

// --- UI-Elemente ---
export const UIElements = {
    fileUploader: document.getElementById('file-uploader'),
    dropZone: document.getElementById('drop-zone'),
    fileListDiv: document.getElementById('file-list'),
    paramsSection: document.getElementById('parameters-section'),
    overlapSlider: document.getElementById('overlap-slider'),
    overlapValue: document.getElementById('overlap-value'),
    mergeCheckbox: document.getElementById('merge-checkbox'),
    startBtn: document.getElementById('start-analysis-btn'),
    analysisProgress: document.getElementById('analysis-progress'),
    progressText: document.getElementById('progress-text'),
    progressBar: document.getElementById('progress-bar'),
    resultsSection: document.getElementById('results-section'),
    resultsTitle: document.getElementById('results-title'),
    resultsSummary: document.getElementById('results-summary'),
    resultsGrid: document.getElementById('results-grid'),
    prognosisSection: document.getElementById('prognosis-section'),
    prognosisTime: document.getElementById('prognosis-time'),
    prognosisSpeed: document.getElementById('prognosis-speed'),
    timeStats: document.getElementById('time-stats'),
};

/**
 * Verarbeitet die vom Benutzer ausgewählten Dateien und zeigt die Prognose an.
 * @param {FileList} files - Die Liste der Dateien.
 * @returns {Promise<{uploadedFiles: Array<File>, fileDurations: Array<number>}>}
 */
export async function handleFiles(files) {
    const uploadedFiles = Array.from(files);
    
    UIElements.prognosisSection.classList.add('hidden');
    UIElements.paramsSection.classList.add('hidden');

    if (uploadedFiles.length > 0) {
        UIElements.fileListDiv.innerHTML = `<h3 class="font-medium text-gray-700">Ausgewählte Dateien:</h3><ul class="list-disc list-inside text-sm text-gray-600">${uploadedFiles.map(f => `<li>${f.name}</li>`).join('')}</ul>`;
        UIElements.paramsSection.classList.remove('hidden');
        UIElements.prognosisTime.textContent = 'berechne...';
        UIElements.prognosisSection.classList.remove('hidden');

        try {
            const getAudioDuration = (file) => new Promise((resolve, reject) => {
                const audio = document.createElement('audio');
                audio.preload = 'metadata';
                audio.onloadedmetadata = () => { URL.revokeObjectURL(audio.src); resolve(audio.duration); };
                audio.onerror = () => { URL.revokeObjectURL(audio.src); reject(`Konnte Dauer für Datei ${file.name} nicht lesen.`); };
                audio.src = URL.createObjectURL(file);
            });

            const fileDurations = await Promise.all(uploadedFiles.map(getAudioDuration));
            const totalDurationSeconds = fileDurations.reduce((sum, d) => sum + d, 0);
            const speedFactor = ESTIMATED_PROCESSING_RATE_MIN_PER_HOUR / 60;
            const estimatedTimeSeconds = speedFactor > 0 ? totalDurationSeconds / speedFactor : Infinity;

            UIElements.prognosisTime.textContent = formatTime(estimatedTimeSeconds);
            UIElements.prognosisSpeed.textContent = `${ESTIMATED_PROCESSING_RATE_MIN_PER_HOUR} min Audio / Stunde`;
            return { uploadedFiles, fileDurations };
        } catch (error) {
            console.error("Fehler beim Lesen der Audio-Metadaten:", error);
            UIElements.prognosisTime.textContent = 'Fehler bei der Berechnung';
            alert(`Ein Fehler ist aufgetreten: ${error}`);
            return { uploadedFiles: [], fileDurations: [] };
        }
    } else {
        UIElements.fileListDiv.innerHTML = '';
        return { uploadedFiles: [], fileDurations: [] };
    }
}

/**
 * Zeichnet ein Spektrogramm auf ein Canvas-Element.
 */
function drawSpectrogram(spec3D, canvas) {
    tf.tidy(() => {
        const spec2D = spec3D.squeeze().transpose();
        const ctx = canvas.getContext('2d');
        const [height, width] = spec2D.shape;
        const specArray = spec2D.arraySync();

        let minVal = Infinity, maxVal = -Infinity;
        specArray.forEach(row => row.forEach(val => {
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }));

        const range = maxVal - minVal;
        const colWidth = canvas.width / width;
        const rowHeight = canvas.height / height;
        
        // Viridis colormap (vereinfacht)
        const viridis = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];
        const getColor = (value) => {
            const i = Math.min(Math.max(Math.floor(value * (viridis.length - 1)), 0), viridis.length - 2);
            const t = value * (viridis.length - 1) - i;
            const r = viridis[i][0] + t * (viridis[i+1][0] - viridis[i][0]);
            const g = viridis[i][1] + t * (viridis[i+1][1] - viridis[i][1]);
            const b = viridis[i][2] + t * (viridis[i+1][2] - viridis[i][2]);
            return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
        };

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let x = 0; x < width; x++) {
            for (let y = 0; y < height; y++) {
                const normalized = (specArray[y][x] - minVal) / range;
                ctx.fillStyle = getColor(normalized);
                ctx.fillRect(x * colWidth, canvas.height - (y + 1) * rowHeight, colWidth, rowHeight);
            }
        }
    });
}

/**
 * Erstellt eine "Ergebnis-Karte" für eine einzelne Erkennung.
 */
function createResultCard(event) {
    const card = document.createElement('div');
    card.className = 'bg-gray-50 p-4 rounded-lg shadow';

    const audio = new Audio();
    audio.src = URL.createObjectURL(createWavBlob(event.audioClip, AUDIO_CONFIG.SAMPLE_RATE));
    audio.controls = true;
    audio.className = 'w-full mt-2';

    const specCanvas = document.createElement('canvas');
    specCanvas.width = 200;
    specCanvas.height = 128;

    card.innerHTML = `
        <p class="text-sm font-medium truncate" title="${event.file}">${event.file}</p>
        <p class="text-xs text-indigo-600 font-bold">Wahrscheinlichkeit: ${(event.prob * 100).toFixed(1)}%</p>
    `;
    card.appendChild(specCanvas);
    card.appendChild(audio);

    // Spektrogramm zeichnen
    const specTensorForViz = audioToMelspectrogram(event.audioClip);
    drawSpectrogram(specTensorForViz, specCanvas);
    tf.dispose(specTensorForViz);

    return card;
}

/**
 * Zeigt die Endergebnisse der Analyse an.
 * @param {Array<Object>} events - Liste aller gefundenen Ereignisse.
 */
export function displayResults(events) {
    UIElements.resultsSection.classList.remove('hidden');
    UIElements.resultsGrid.innerHTML = '';
    UIElements.resultsSummary.innerHTML = '';
    UIElements.resultsTitle.textContent = `2. Ergebnisse: ${events.length} mögliche Uhu-Rufe gefunden`;

    if (events.length === 0) {
        UIElements.resultsSummary.innerHTML = `<div class="bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 rounded-md" role="alert"><p>Es wurden keine Ereignisse über dem Schwellenwert von 50% gefunden.</p></div>`;
        return;
    }

    const highConfidenceEvents = events.filter(e => e.prob >= 0.98).sort((a, b) => b.prob - a.prob);
    const mediumConfidenceEvents = events.filter(e => e.prob >= 0.5 && e.prob < 0.98).sort((a, b) => b.prob - a.prob);

    const appendClips = (eventList, titleText, titleColor) => {
        if (eventList.length > 0) {
            const title = document.createElement('h3');
            title.className = `col-span-full text-xl font-bold text-gray-800 mt-6 mb-2 ${titleColor}`;
            const clipsToShow = eventList.slice(0, MAX_CLIPS_TO_SHOW_PER_CATEGORY);
            title.textContent = `${titleText} (Top ${clipsToShow.length})`;
            UIElements.resultsGrid.appendChild(title);
            clipsToShow.forEach(event => UIElements.resultsGrid.appendChild(createResultCard(event)));
        }
    };

    appendClips(highConfidenceEvents, '✅ Uhu erkannt mit hoher Wahrscheinlichkeit', 'text-green-700');
    appendClips(mediumConfidenceEvents, '🤔 Potentiell Uhu erkannt, bitte prüfen', 'text-yellow-700');
}