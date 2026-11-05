import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, AlertOctagon, HelpCircle } from 'lucide-react';
import { clsx } from 'clsx';

export interface GaugeProps {
  probability: number; // 0.0 to 1.0 (calibrated P(genuine))
  verdict: 'BONA FIDE (GENUINE)' | 'SPOOF (SYNTHETIC)' | 'INCONCLUSIVE';
  threshold?: number;
  className?: string;
  size?: number;
}

export const Gauge: React.FC<GaugeProps> = ({
  probability,
  verdict,
  threshold = 0.1795,
  className,
  size = 180,
}) => {
  const percentage = Math.min(100, Math.max(0, probability * 100));
  const isGenuine = verdict === 'BONA FIDE (GENUINE)';
  const isSpoof = verdict === 'SPOOF (SYNTHETIC)';
  const isUncertain = verdict === 'INCONCLUSIVE' || (percentage >= 40 && percentage <= 60);

  // Circular progress geometry
  const strokeWidth = 8;
  const radius = (size - strokeWidth * 2) / 2;
  const center = size / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (circumference * percentage) / 100;

  const statusConfig = isGenuine
    ? {
        label: 'Authentic Human Voice',
        tag: 'Bona Fide',
        subtext: 'Acoustic resonance and pitch modulation match natural speech',
        color: 'var(--status-genuine)',
        textColor: 'text-genuine',
        pillStyle: 'bg-genuine-bg text-genuine border-genuine-border',
        icon: <CheckCircle2 className="w-4 h-4 text-genuine" />,
      }
    : isSpoof
    ? {
        label: 'Synthetic Voice Detected',
        tag: 'Deepfake',
        subtext: 'Neural vocoder phase anomalies and synthesis artifacts identified',
        color: 'var(--status-spoof)',
        textColor: 'text-spoof',
        pillStyle: 'bg-spoof-bg text-spoof border-spoof-border',
        icon: <AlertOctagon className="w-4 h-4 text-spoof" />,
      }
    : {
        label: 'Inconclusive Voice',
        tag: 'Uncertain',
        subtext: 'Score falls within calibrated uncertainty interval [40% - 60%]',
        color: 'var(--status-uncertain)',
        textColor: 'text-uncertain',
        pillStyle: 'bg-uncertain-bg text-uncertain border-uncertain-border',
        icon: <HelpCircle className="w-4 h-4 text-uncertain" />,
      };

  return (
    <div
      className={clsx(
        'p-5 sm:p-6 rounded-[12px] bg-surface border border-borderSubtle shadow-card flex flex-col items-center text-center justify-between transition-colors duration-150',
        className
      )}
    >
      {/* Top Status Pill */}
      <div className={clsx('px-3 py-1 rounded-full border text-xs font-medium flex items-center gap-1.5', statusConfig.pillStyle)}>
        {statusConfig.icon}
        <span>{statusConfig.tag}</span>
      </div>

      {/* Center Circular Progress Ring */}
      <div className="relative my-3 flex items-center justify-center">
        <svg width={size} height={size} className="transform -rotate-90">
          {/* Background Track */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="transparent"
            stroke="var(--border-subtle)"
            strokeWidth={strokeWidth}
          />
          {/* Animated Value Arc */}
          <motion.circle
            cx={center}
            cy={center}
            r={radius}
            fill="transparent"
            stroke={statusConfig.color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            strokeLinecap="round"
          />
        </svg>

        {/* Center Percentage Display */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <span className="text-3xl font-semibold font-mono tracking-tight text-txt-primary tabular-nums">
            {percentage.toFixed(1)}%
          </span>
          <span className="text-xs uppercase tracking-wide text-txt-secondary font-medium mt-0.5">
            Confidence
          </span>
        </div>
      </div>

      {/* Bottom Summary Text */}
      <div className="w-full text-center">
        <h3 className={clsx('text-base font-semibold tracking-tight', statusConfig.textColor)}>
          {statusConfig.label}
        </h3>
        <p className="text-xs text-txt-secondary mt-1 max-w-[240px] mx-auto leading-relaxed">
          {statusConfig.subtext}
        </p>
      </div>
    </div>
  );
};
