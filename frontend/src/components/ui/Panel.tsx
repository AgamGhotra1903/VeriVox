import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export interface PanelProps {
  title?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  recessed?: boolean;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  badge,
  actions,
  children,
  className,
  recessed = false,
}) => {
  return (
    <section
      className={twMerge(
        'border border-borderSubtle rounded-[12px] overflow-hidden flex flex-col shadow-card animate-card-mount transition-colors duration-150',
        recessed ? 'bg-inset' : 'bg-surface',
        className
      )}
    >
      {title && (
        <header className="px-5 sm:px-6 py-4 border-b border-borderSubtle flex items-center justify-between select-none">
          <div className="flex items-center gap-2.5">
            <h2 className="text-base font-semibold text-txt-primary tracking-tight">
              {title}
            </h2>
            {badge && <span className="ml-1">{badge}</span>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </header>
      )}

      <div className="p-5 sm:p-6 flex-1 relative">
        {children}
      </div>
    </section>
  );
};
