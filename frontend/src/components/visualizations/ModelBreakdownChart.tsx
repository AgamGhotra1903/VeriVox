import React from 'react';
import { motion } from 'framer-motion';
import { Check, X, ShieldAlert, ShieldCheck } from 'lucide-react';
import { clsx } from 'clsx';
import { ModelScoreItem } from '../../types/api';

export interface ModelBreakdownChartProps {
  models: ModelScoreItem[];
  fusedScore: number;
  finalVerdict: string;
  className?: string;
}

export const ModelBreakdownChart: React.FC<ModelBreakdownChartProps> = ({
  models,
  fusedScore,
  finalVerdict,
  className,
}) => {
  const maxAbsScore = Math.max(
    ...models.map((m) => Math.abs(m.score)),
    Math.abs(fusedScore),
    1.0
  );

  const isFinalGenuine = finalVerdict.includes('GENUINE');

  return (
    <div
      className={clsx(
        'border border-borderSubtle bg-surface rounded-[12px] p-5 sm:p-6 flex flex-col gap-4 shadow-card select-none animate-card-mount',
        className
      )}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-borderSubtle pb-3">
        <div>
          <h3 className="text-base font-semibold text-txt-primary">
            Model Ensemble Breakdown
          </h3>
          <p className="text-xs text-txt-secondary mt-0.5">
            Individual decisions across all 6 base models and stacked meta-learner
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs text-txt-secondary">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-spoof" />
            Negative = Synthetic
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-genuine" />
            Positive = Genuine
          </span>
        </div>
      </div>

      <div className="space-y-2">
        {models.map((model) => {
          const isModelGenuine = model.score > 0;
          const agrees = isModelGenuine === isFinalGenuine;
          const scorePercent = Math.min(100, (Math.abs(model.score) / maxAbsScore) * 100);

          return (
            <div
              key={model.name}
              className="grid grid-cols-[160px_70px_1fr_90px_24px] items-center gap-3 text-xs py-2 px-3 rounded-[8px] bg-elevated/40 border border-borderSubtle hover:bg-elevated/70 transition-colors duration-150"
            >
              {}
              <span className="font-medium text-txt-primary truncate" title={model.name}>
                {model.name}
              </span>

              {}
              <span className="text-xs font-mono text-txt-secondary tabular-nums">
                {model.ensemble_weight > 0 ? `+${model.ensemble_weight.toFixed(3)}` : model.ensemble_weight.toFixed(3)}
              </span>

              {}
              <div className="h-3.5 bg-inset rounded-[4px] relative flex items-center overflow-hidden border border-borderSubtle">
                <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-borderStrong z-10" />

                {}
                {model.score < 0 ? (
                  <div
                    className="absolute right-1/2 top-0 bottom-0 bg-spoof rounded-l"
                    style={{ width: `${scorePercent / 2}%` }}
                  />
                ) : (
                  <div
                    className="absolute left-1/2 top-0 bottom-0 bg-genuine rounded-r"
                    style={{ width: `${scorePercent / 2}%` }}
                  />
                )}
              </div>

              {}
              <span className="text-right font-mono text-xs tabular-nums text-txt-primary">
                {model.score > 0 ? `+${model.score.toFixed(3)}` : model.score.toFixed(3)}
              </span>

              {}
              <div className="flex justify-center">
                {agrees ? (
                  <span title="Agrees with final ensemble">
                    <Check className="w-3.5 h-3.5 text-genuine stroke-[2]" />
                  </span>
                ) : (
                  <span title="Disagrees with final ensemble">
                    <X className="w-3.5 h-3.5 text-txt-muted" />
                  </span>
                )}
              </div>
            </div>
          );
        })}

        {}
        <div className="grid grid-cols-[160px_70px_1fr_90px_24px] items-center gap-3 text-xs py-2.5 px-3 rounded-[8px] bg-primary/5 border border-primary/20 shadow-sm mt-3 font-semibold">
          <span className="text-primary font-semibold">
            Stacked Meta-Learner
          </span>

          <span className="text-xs font-mono text-primary font-medium">
            Fused
          </span>

          <div className="h-3.5 bg-inset rounded-[4px] relative flex items-center overflow-hidden border border-borderSubtle">
            <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-borderStrong z-10" />
            {fusedScore < 0 ? (
              <div
                className="absolute right-1/2 top-0 bottom-0 bg-spoof rounded-l"
                style={{ width: `${Math.min(100, (Math.abs(fusedScore) / maxAbsScore) * 100) / 2}%` }}
              />
            ) : (
              <div
                className="absolute left-1/2 top-0 bottom-0 bg-primary rounded-r"
                style={{ width: `${Math.min(100, (Math.abs(fusedScore) / maxAbsScore) * 100) / 2}%` }}
              />
            )}
          </div>

          <span className="text-right font-mono text-xs tabular-nums text-primary font-semibold">
            {fusedScore > 0 ? `+${fusedScore.toFixed(3)}` : fusedScore.toFixed(3)}
          </span>

          <div className="flex justify-center">
            {isFinalGenuine ? (
              <ShieldCheck className="w-4 h-4 text-genuine" />
            ) : (
              <ShieldAlert className="w-4 h-4 text-spoof" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
