// --- Hilfsfunktionen ---

/**
 * Formatiert eine Gesamtsekundenzahl in ein menschenlesbares Format (z.B. "ca. 5 Minuten").
 * @param {number} totalSeconds - Die zu formatierende Sekundenzahl.
 * @returns {string} Der formatierte Zeit-String.
 */
export function formatTime(totalSeconds) {
    if (isNaN(totalSeconds) || totalSeconds < 0) return 'wird berechnet...';
    if (totalSeconds < 1) return "weniger als eine Sekunde";
    if (totalSeconds < 60) return `ca. ${Math.round(totalSeconds)} Sekunden`;
    if (totalSeconds < 3600) return `ca. ${Math.round(totalSeconds / 60)} Minuten`;

    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.round((totalSeconds % 3600) / 60);

    if (minutes === 0) return `ca. ${hours} Stunde${hours > 1 ? 'n' : ''}`;
    if (minutes === 60) return `ca. ${hours + 1} Stunde${hours + 1 > 1 ? 'n' : ''}`;
    return `ca. ${hours} Stunde${hours > 1 ? 'n' : ''} und ${minutes} Minuten`;
}

/**
 * Formatiert die verbleibende Zeit.
 * @param {number} seconds - Verbleibende Sekunden.
 * @returns {string} Formatierter String.
 */
export function formatRemainingTime(seconds) {
    if (isNaN(seconds) || seconds < 0 || seconds === Infinity) return 'Verbleibend: berechne...';
    if (seconds < 2) return 'Verbleibend: fast fertig...';
    
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    
    if (h > 0) return `Verbleibend: ~${h} Std. ${m} Min.`;
    if (m > 0) return `Verbleibend: ~${m} Min. ${s} Sek.`;
    return `Verbleibend: ~${s} Sek.`;
}

/**
 * Erstellt ein WAV-Blob aus rohen Audiodaten.
 * @param {Float32Array} samples - Die Audiodaten.
 * @param {number} sampleRate - Die Abtastrate.
 * @returns {Blob} Ein Blob im WAV-Format.
 */
export function createWavBlob(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

/**
 * Fügt überlappende Erkennungsereignisse zusammen.
 * @param {Array<Object>} candidates - Eine Liste von Erkennungen.
 * @returns {Array<Object>} Die zusammengeführte Liste.
 */
export function mergeEvents(candidates) {
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