import React from 'react';
import { clsx } from 'clsx';

export interface ReadoutProps {
  label: string;
  value: string | number;
  unit?: string;
  sublabel?: string;
  indicatorColor?: 'genuine' | 'spoof' | 'uncertain' | 'neutral';
  className?: string;
}

export const Readout: React.FC<ReadoutProps> = ({
  label,
  value,
  unit,
  sublabel,
  indicatorColor = 'neutral',
  className,
}) => {
  const indicatorMap = {
    genuine: 'bg-genuine',
    spoof: 'bg-spoof',
    uncertain: 'bg-uncertain',
    neutral: 'bg-txt-muted',
  };

  return (
    <div
      className={clsx(
        'p-4 bg-surface border border-borderSubtle rounded-[12px] shadow-card flex flex-col justify-between transition-colors duration-150 hover:border-borderStrong',
        className
      )}
    >
      <div className="flex items-center justify-between gap-1 mb-2">
        <span className="text-xs uppercase tracking-wide font-medium text-txt-secondary">
          {label}
        </span>
        <span
          className={clsx(
            'w-2 h-2 rounded-full flex-shrink-0',
            indicatorMap[indicatorColor]
          )}
        />
      </div>

      <div className="flex items-baseline gap-1.5">
        <span className="text-2xl font-semibold font-mono tracking-tight text-txt-primary tabular-nums">
          {value}
        </span>
        {unit && (
          <span className="text-xs font-medium font-sans text-txt-secondary">
            {unit}
          </span>
        )}
      </div>

      {sublabel && (
        <span className="text-xs text-txt-secondary mt-1.5 truncate">
          {sublabel}
        </span>
      )}
    </div>
  );
};
