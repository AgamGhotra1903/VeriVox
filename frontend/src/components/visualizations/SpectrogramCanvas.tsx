import React, { useRef, useEffect, useState } from 'react';
import { clsx } from 'clsx';

export interface SpectrogramCanvasProps {
  lfccData: number[][]; // (60, T)
  logmelData: number[][]; // (80, T)
  durationSec: number;
  currentTime?: number;
  height?: number;
  className?: string;
}

export const SpectrogramCanvas: React.FC<SpectrogramCanvasProps> = ({
  lfccData,
  logmelData,
  durationSec,
  currentTime = 0,
  height = 120,
  className,
}) => {
  const [activeMode, setActiveMode] = useState<'lfcc' | 'logmel'>('lfcc');
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const activeMatrix = activeMode === 'lfcc' ? lfccData : logmelData;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !activeMatrix || activeMatrix.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const isDark =
      typeof window !== 'undefined' &&
      (document.documentElement.classList.contains('dark') ||
        window.matchMedia('(prefers-color-scheme: dark)').matches);

    const numFreqBins = activeMatrix.length;
    const numTimeBins = activeMatrix[0].length;
    if (numTimeBins === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = height;

    // Find min and max for normalization
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let f = 0; f < numFreqBins; f++) {
      for (let t = 0; t < numTimeBins; t++) {
        const v = activeMatrix[f][t];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
    const valRange = maxVal - minVal || 1e-6;

    const binW = w / numTimeBins;
    const binH = h / numFreqBins;

    // Restrained, calm sequential colormap (Deep Slate -> Muted Indigo -> Soft Teal -> Pale Gold)
    for (let f = 0; f < numFreqBins; f++) {
      const y = h - (f + 1) * binH;
      for (let t = 0; t < numTimeBins; t++) {
        const norm = (activeMatrix[f][t] - minVal) / valRange;
        const x = t * binW;

        let r = 0, g = 0, b = 0;
        if (norm < 0.35) {
          const factor = norm / 0.35;
          r = Math.round(15 + 45 * factor);
          g = Math.round(23 + 40 * factor);
          b = Math.round(42 + 90 * factor);
        } else if (norm < 0.75) {
          const factor = (norm - 0.35) / 0.4;
          r = Math.round(60 + 50 * factor);
          g = Math.round(63 + 90 * factor);
          b = Math.round(132 + 30 * factor);
        } else {
          const factor = (norm - 0.75) / 0.25;
          r = Math.round(110 + 110 * factor);
          g = Math.round(153 + 70 * factor);
          b = Math.round(162 - 40 * factor);
        }

        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x, y, Math.ceil(binW), Math.ceil(binH));
      }
    }

    // Playhead line
    if (durationSec > 0) {
      const progressFrac = currentTime / durationSec;
      const playheadX = progressFrac * w;
      ctx.strokeStyle = isDark ? '#ffffff' : '#0f172a';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, h);
      ctx.stroke();
    }
  }, [activeMatrix, durationSec, currentTime, height]);

  return (
    <div className={clsx('flex flex-col gap-2', className)}>
      {/* Mode Switcher Bar */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-txt-secondary uppercase tracking-wide">
          Spectrogram Representation
        </span>

        <div className="flex items-center gap-1 p-0.5 bg-elevated border border-borderSubtle rounded-[8px]">
          <button
            onClick={() => setActiveMode('lfcc')}
            className={clsx(
              'px-2.5 py-1 rounded-[6px] text-xs transition-colors duration-150',
              activeMode === 'lfcc'
                ? 'bg-surface text-txt-primary font-medium shadow-card'
                : 'text-txt-secondary hover:text-txt-primary'
            )}
          >
            LFCC (60-dim)
          </button>
          <button
            onClick={() => setActiveMode('logmel')}
            className={clsx(
              'px-2.5 py-1 rounded-[6px] text-xs transition-colors duration-150',
              activeMode === 'logmel'
                ? 'bg-surface text-txt-primary font-medium shadow-card'
                : 'text-txt-secondary hover:text-txt-primary'
            )}
          >
            Log-Mel (80-dim)
          </button>
        </div>
      </div>

      {/* Canvas Well with Frequency Axis Ticks */}
      <div className="relative rounded-[8px] overflow-hidden border border-borderSubtle bg-chassis">
        <canvas ref={canvasRef} height={height} className="w-full block select-none" />

        {/* Y-Axis Frequency Indicators */}
        <div className="absolute top-1.5 left-2 pointer-events-none text-[10px] font-mono text-txt-secondary bg-surface/90 px-1.5 py-0.5 rounded border border-borderSubtle">
          {activeMode === 'lfcc' ? 'Coeff 60 (HF)' : '8 kHz'}
        </div>
        <div className="absolute bottom-1.5 left-2 pointer-events-none text-[10px] font-mono text-txt-secondary bg-surface/90 px-1.5 py-0.5 rounded border border-borderSubtle">
          {activeMode === 'lfcc' ? 'Coeff 1 (LF)' : '0 Hz'}
        </div>
      </div>
    </div>
  );
};
