import React, { useState, useRef } from 'react';

export interface AttentionHeatstripProps {
  attentionWeights: number[]; // normalized [0, 1] per frame
  durationSec: number;
  currentTime?: number;
  className?: string;
  height?: number;
}

export const AttentionHeatstrip: React.FC<AttentionHeatstripProps> = ({
  attentionWeights,
  durationSec,
  currentTime,
  className,
  height = 18,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    timeSec: number;
    weight: number;
  } | null>(null);

  if (!attentionWeights || attentionWeights.length === 0) {
    return null;
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const frameIndex = Math.min(
      attentionWeights.length - 1,
      Math.floor(ratio * attentionWeights.length)
    );
    setHoverInfo({
      x: e.clientX - rect.left,
      timeSec: ratio * durationSec,
      weight: attentionWeights[frameIndex],
    });
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  return (
    <div className={`flex flex-col gap-1.5 select-none ${className}`}>
      <div className="flex items-center justify-between text-xs text-txt-secondary">
        <span className="flex items-center gap-1.5 font-medium text-txt-primary">
          <span className="w-2 h-2 rounded-full bg-primary inline-block" />
          Attention Salience (Frames Influencing Decision)
        </span>
        <span className="text-xs text-txt-muted">
          Hover to inspect frame weight
        </span>
      </div>

      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="relative border border-borderSubtle bg-elevated/40 rounded-[8px] overflow-hidden cursor-crosshair"
        style={{ height }}
      >
        {/* Render horizontal gradient heatstrip with calm indigo palette */}
        <div className="w-full h-full flex">
          {attentionWeights.map((w, idx) => {
            const alpha = Math.max(0.1, w);
            return (
              <div
                key={idx}
                className="flex-1 h-full"
                style={{
                  backgroundColor:
                    w > 0.65
                      ? `rgba(217, 119, 6, ${alpha})`
                      : `rgba(99, 102, 241, ${alpha})`,
                }}
              />
            );
          })}
        </div>

        {/* Hover Tooltip Overlay with rounded tooltip */}
        {hoverInfo && (
          <div
            className="absolute top-0 bottom-0 pointer-events-none border-l border-txt-primary shadow-sm"
            style={{ left: hoverInfo.x }}
          >
            <div className="absolute -top-7 -translate-x-1/2 bg-surface border border-borderSubtle text-txt-primary px-2 py-0.5 rounded-[6px] text-xs font-mono whitespace-nowrap shadow-card">
              {hoverInfo.timeSec.toFixed(2)}s · Weight: {(hoverInfo.weight * 100).toFixed(1)}%
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
