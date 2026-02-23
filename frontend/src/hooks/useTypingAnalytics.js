import { useCallback, useRef, useState } from 'react';

/**
 * useTypingAnalytics — Captures real-time typing dynamics for CERAS.
 * 
 * Tracks: keystrokes, WPM, pause/hesitation events, deletion ratio,
 * and session duration. All metrics map directly to CERAS features
 * (keystrokes, formulation_time, behavioral signals).
 */
export default function useTypingAnalytics() {
  const [analytics, setAnalytics] = useState({
    totalKeystrokes: 0,
    deletions: 0,
    deletionRatio: 0,
    wpm: 0,
    hesitations: 0,        // pauses > 2000ms
    longestPause: 0,       // ms
    currentPause: 0,       // ms (live)
    isHesitating: false,
    sessionDuration: 0,    // seconds
    avgKeystrokeInterval: 0,
  });

  const stateRef = useRef({
    totalKeys: 0,
    delKeys: 0,
    wordCount: 0,
    timestamps: [],           // last N keystroke timestamps
    lastKeystrokeTime: null,
    sessionStart: null,
    hesitationCount: 0,
    longestPause: 0,
    pauseTimerId: null,
    isHesitating: false,
  });

  // Call this on every keydown in the textarea
  const onKeyDown = useCallback((e) => {
    const now = Date.now();
    const s = stateRef.current;

    // Start session timer on first keystroke
    if (!s.sessionStart) {
      s.sessionStart = now;
    }

    // Track keystroke timestamps (keep last 30 for WPM calc)
    s.timestamps.push(now);
    if (s.timestamps.length > 30) s.timestamps.shift();

    // Deletion tracking
    const isDeletion = e.key === 'Backspace' || e.key === 'Delete';
    s.totalKeys++;
    if (isDeletion) s.delKeys++;

    // Pause / hesitation detection
    if (s.lastKeystrokeTime) {
      const gap = now - s.lastKeystrokeTime;
      if (gap > s.longestPause) s.longestPause = gap;
      if (gap > 2000) {
        s.hesitationCount++;
      }
    }
    s.lastKeystrokeTime = now;
    s.isHesitating = false;

    // Clear and restart hesitation timer
    if (s.pauseTimerId) clearTimeout(s.pauseTimerId);
    s.pauseTimerId = setTimeout(() => {
      s.isHesitating = true;
      updateState();
    }, 2000);

    updateState();
  }, []);

  // Compute derived metrics and push to React state
  const updateState = useCallback(() => {
    const s = stateRef.current;
    const now = Date.now();
    const sessionSec = s.sessionStart ? (now - s.sessionStart) / 1000 : 0;

    // WPM: use recent keystroke timestamps
    let wpm = 0;
    if (s.timestamps.length >= 2) {
      const span = (s.timestamps[s.timestamps.length - 1] - s.timestamps[0]) / 1000 / 60;
      if (span > 0) {
        // Approximate: 5 chars per word
        wpm = Math.round((s.timestamps.length / 5) / span);
      }
    }

    // Avg keystroke interval
    let avgInterval = 0;
    if (s.timestamps.length >= 2) {
      const intervals = [];
      for (let i = 1; i < s.timestamps.length; i++) {
        intervals.push(s.timestamps[i] - s.timestamps[i - 1]);
      }
      avgInterval = Math.round(intervals.reduce((a, b) => a + b, 0) / intervals.length);
    }

    const currentPause = s.lastKeystrokeTime ? now - s.lastKeystrokeTime : 0;

    setAnalytics({
      totalKeystrokes: s.totalKeys,
      deletions: s.delKeys,
      deletionRatio: s.totalKeys > 0 ? s.delKeys / s.totalKeys : 0,
      wpm: Math.min(wpm, 200),
      hesitations: s.hesitationCount,
      longestPause: s.longestPause,
      currentPause,
      isHesitating: s.isHesitating,
      sessionDuration: Math.round(sessionSec),
      avgKeystrokeInterval: avgInterval,
    });
  }, []);

  // Simulate realistic analytics for pasted / pre-filled text at 50 WPM avg
  const simulateFromPaste = useCallback((text) => {
    const words = text.trim().split(/\s+/).filter(Boolean);
    const wordCount = words.length;
    const charCount = text.length;
    const AVG_WPM = 50;
    const AVG_INTERVAL = Math.round(60000 / (AVG_WPM * 5)); // ~240ms
    const sessionSec = wordCount > 0 ? Math.round((wordCount / AVG_WPM) * 60) : 0;

    const s = stateRef.current;
    if (s.pauseTimerId) clearTimeout(s.pauseTimerId);

    // Update internal state so subsequent typed keys build on top
    const now = Date.now();
    stateRef.current = {
      totalKeys: charCount,
      delKeys: 0,
      wordCount,
      timestamps: [now - AVG_INTERVAL, now], // minimal pair for future WPM calc
      lastKeystrokeTime: now,
      sessionStart: now - sessionSec * 1000,
      hesitationCount: 0,
      longestPause: 500,
      pauseTimerId: null,
      isHesitating: false,
    };

    setAnalytics({
      totalKeystrokes: charCount,
      deletions: 0,
      deletionRatio: 0,
      wpm: AVG_WPM,
      hesitations: 0,
      longestPause: 500,
      currentPause: 0,
      isHesitating: false,
      sessionDuration: sessionSec,
      avgKeystrokeInterval: AVG_INTERVAL,
    });
  }, []);

  // Reset all analytics (for "New Problem")
  const reset = useCallback(() => {
    const s = stateRef.current;
    if (s.pauseTimerId) clearTimeout(s.pauseTimerId);
    stateRef.current = {
      totalKeys: 0,
      delKeys: 0,
      wordCount: 0,
      timestamps: [],
      lastKeystrokeTime: null,
      sessionStart: null,
      hesitationCount: 0,
      longestPause: 0,
      pauseTimerId: null,
      isHesitating: false,
    };
    setAnalytics({
      totalKeystrokes: 0,
      deletions: 0,
      deletionRatio: 0,
      wpm: 0,
      hesitations: 0,
      longestPause: 0,
      currentPause: 0,
      isHesitating: false,
      sessionDuration: 0,
      avgKeystrokeInterval: 0,
    });
  }, []);

  return { analytics, onKeyDown, simulateFromPaste, reset };
}
