import React from 'react';
import { motion } from 'framer-motion';
import { Check, Loader2, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { PipelineEvent } from '../../types/api';

export interface PipelineTrackerProps {
  events: PipelineEvent[];
  isComplete: boolean;
  error?: string | null;
  className?: string;
}

export const PipelineTracker: React.FC<PipelineTrackerProps> = ({
  events,
  isComplete,
  error,
  className,
}) => {
  const stageDefs = [
    { key: 'validating', label: 'Audio Validation', desc: '16 kHz mono check, SNR, and clipping estimation' },
    { key: 'resampling', label: 'Window Alignment', desc: '4.0s (64,000 samples) center crop / padding' },
    { key: 'feature_extraction', label: 'Feature Extraction', desc: 'LFCC (60-dim) and Log-Mel (80-dim) filterbanks' },
    { key: 'model_scores', label: 'Ensemble Evaluation', desc: 'CRNNs, GMM, SVM, Random Forest, Logistic Reg' },
    { key: 'ensemble', label: 'Meta-Learner Stacking', desc: 'L2 logistic regression fusion & calibration' },
  ];

  const latestEvent = events.length > 0 ? events[events.length - 1] : null;
  const currentStageKey = latestEvent ? latestEvent.stage : null;

  const stageTimings: Record<string, number> = {};
  events.forEach((ev) => {
    stageTimings[ev.stage] = ev.elapsed_ms;
  });

  return (
    <div
      className={clsx(
        'border border-borderSubtle bg-surface rounded-[12px] p-4 sm:p-5 flex flex-col gap-3 shadow-card select-none animate-card-mount',
        className
      )}
    >
      <div className="flex items-center justify-between border-b border-borderSubtle pb-3">
        <div className="flex items-center gap-2">
          <span
            className={clsx(
              'w-2 h-2 rounded-full inline-block',
              error
                ? 'bg-spoof'
                : isComplete
                ? 'bg-genuine'
                : 'bg-primary animate-pulse'
            )}
          />
          <span className="text-xs font-semibold text-txt-primary">
            Analysis Pipeline
          </span>
        </div>
        {latestEvent && (
          <span className="text-xs font-mono text-txt-secondary tabular-nums">
            {latestEvent.elapsed_ms.toFixed(0)} ms
          </span>
        )}
      </div>

      <div className="space-y-2">
        {stageDefs.map((stage) => {
          const isDone = isComplete || (events.some((e) => e.stage === stage.key) && currentStageKey !== stage.key);
          const isCurrent = currentStageKey === stage.key && !isComplete;
          const timing = stageTimings[stage.key];

          return (
            <div
              key={stage.key}
              className={clsx(
                'flex items-center justify-between p-2.5 rounded-[8px] border text-xs transition-colors duration-150',
                isDone
                  ? 'bg-surface border-borderSubtle text-txt-primary'
                  : isCurrent
                  ? 'bg-primary/5 border-primary/20 text-txt-primary'
                  : 'bg-elevated/40 border-transparent text-txt-muted'
              )}
            >
              <div className="flex items-center gap-2.5">
                <div
                  className={clsx(
                    'w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 text-[10px]',
                    isDone
                      ? 'bg-genuine/10 text-genuine'
                      : isCurrent
                      ? 'bg-primary/10 text-primary'
                      : 'bg-elevated text-txt-muted'
                  )}
                >
                  {isDone ? (
                    <Check className="w-3 h-3 stroke-[2.5]" />
                  ) : isCurrent ? (
                    <Loader2 className="w-3 h-3 animate-spin text-primary" />
                  ) : (
                    '·'
                  )}
                </div>

                <div>
                  <p className="font-medium text-txt-primary">
                    {stage.label}
                  </p>
                  <p className="text-[11px] text-txt-secondary">
                    {stage.desc}
                  </p>
                </div>
              </div>

              {timing !== undefined && (
                <span className="text-[11px] font-mono text-txt-secondary tabular-nums ml-2">
                  {timing.toFixed(0)} ms
                </span>
              )}
            </div>
          );
        })}
      </div>

      {error && (
        <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 dark:bg-rose-950/20 dark:border-rose-900/40 dark:text-rose-300 rounded-[8px] text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-spoof flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
