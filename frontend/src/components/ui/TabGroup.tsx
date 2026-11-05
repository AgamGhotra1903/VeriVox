import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export interface TabItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  count?: number;
}

export interface TabGroupProps {
  tabs: TabItem[];
  activeId: string;
  onChange: (id: string) => void;
  className?: string;
  size?: 'sm' | 'md';
}

export const TabGroup: React.FC<TabGroupProps> = ({
  tabs,
  activeId,
  onChange,
  className,
  size = 'md',
}) => {
  return (
    <div
      role="tablist"
      className={twMerge(
        'inline-flex bg-elevated p-1 border border-borderSubtle rounded-[10px] select-none',
        className
      )}
    >
      {tabs.map((tab) => {
        const isActive = tab.id === activeId;
        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(tab.id)}
            className={clsx(
              'font-medium transition-colors duration-150 flex items-center gap-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 rounded-[8px] min-h-[36px]',
              size === 'sm' ? 'px-3 py-1.5 text-xs' : 'px-4 py-2 text-sm',
              isActive
                ? 'bg-surface text-txt-primary font-medium shadow-card border border-borderSubtle'
                : 'text-txt-secondary hover:text-txt-primary hover:bg-surface/50 border border-transparent'
            )}
          >
            {tab.icon && <span className="flex-shrink-0">{tab.icon}</span>}
            <span>{tab.label}</span>
            {tab.count !== undefined && (
              <span
                className={clsx(
                  'text-[11px] px-1.5 py-0.5 rounded-full font-mono',
                  isActive ? 'bg-primary/10 text-primary font-medium' : 'bg-elevated text-txt-secondary'
                )}
              >
                {tab.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
