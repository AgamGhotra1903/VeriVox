import React, { useRef, useEffect } from 'react';

export interface WaveformCanvasProps {
  peaks: number[]; // Min/Max downsampled peak pairs
  durationSec: number;
  currentTime: number;
  onSeek?: (time: number) => void;
  height?: number;
  className?: string;
}

export const WaveformCanvas: React.FC<WaveformCanvasProps> = ({
  peaks,
  durationSec,
  currentTime,
  onSeek,
  height = 96,
  className,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDraggingRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || peaks.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const isDark =
      typeof window !== 'undefined' &&
      (document.documentElement.classList.contains('dark') ||
        window.matchMedia('(prefers-color-scheme: dark)').matches);

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = height;
    const midY = h / 2;

    // Neutral background fill
    ctx.fillStyle = isDark ? '#0f172a' : '#f8fafc';
    ctx.fillRect(0, 0, w, h);

    // Light center baseline
    ctx.strokeStyle = isDark ? 'rgba(148, 163, 184, 0.15)' : 'rgba(148, 163, 184, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    // Light time gridlines
    const numMarkers = 6;
    ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
    ctx.font = '10px Inter, sans-serif';
    for (let i = 1; i < numMarkers; i++) {
      const x = (w / numMarkers) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.strokeStyle = isDark ? 'rgba(148, 163, 184, 0.1)' : 'rgba(148, 163, 184, 0.15)';
      ctx.stroke();

      const sec = ((durationSec / numMarkers) * i).toFixed(1);
      ctx.fillText(`${sec}s`, x + 4, h - 6);
    }

    // Draw waveform bars with restrained palette
    const numPairs = peaks.length / 2;
    const barWidth = Math.max(1.5, w / numPairs);
    const progressFrac = durationSec > 0 ? currentTime / durationSec : 0;
    const playheadX = progressFrac * w;

    const playedColor = isDark ? '#6366f1' : '#4f46e5'; // Muted indigo
    const unplayedColor = isDark ? '#334155' : '#cbd5e1'; // Neutral slate

    for (let i = 0; i < numPairs; i++) {
      const minVal = peaks[i * 2];
      const maxVal = peaks[i * 2 + 1];
      const x = i * barWidth;

      const yMin = midY - maxVal * (midY * 0.88);
      const yMax = midY - minVal * (midY * 0.88);
      const barHeight = Math.max(2, yMax - yMin);

      ctx.fillStyle = x <= playheadX ? playedColor : unplayedColor;
      ctx.beginPath();
      ctx.roundRect
        ? ctx.roundRect(x, yMin, Math.max(1, barWidth - 1), barHeight, 2)
        : ctx.rect(x, yMin, Math.max(1, barWidth - 1), barHeight);
      ctx.fill();
    }

    // Minimal playhead line
    if (durationSec > 0) {
      ctx.strokeStyle = isDark ? '#f8fafc' : '#0f172a';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, h);
      ctx.stroke();

      ctx.fillStyle = playedColor;
      ctx.beginPath();
      ctx.arc(playheadX, 4, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [peaks, durationSec, currentTime, height]);

  // Click & drag scrub handling
  const handleSeekEvent = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek || !canvasRef.current || durationSec <= 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const clickX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const targetTime = (clickX / rect.width) * durationSec;
    onSeek(targetTime);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDraggingRef.current = true;
    handleSeekEvent(e);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDraggingRef.current) {
      handleSeekEvent(e);
    }
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  return (
    <div className="relative rounded-[8px] overflow-hidden border border-borderSubtle bg-chassis group">
      <canvas
        ref={canvasRef}
        height={height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="w-full cursor-pointer block select-none"
        title="Click or drag to scrub playback"
      />
    </div>
  );
};
