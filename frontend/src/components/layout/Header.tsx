import React from 'react';
import { Activity, Sliders, BarChart3, BookOpen, Printer, Shield } from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from '../ui/Button';

export interface HeaderProps {
  activeScreen: 'analyse' | 'stress-test' | 'benchmarks' | 'method';
  onScreenChange: (screen: 'analyse' | 'stress-test' | 'benchmarks' | 'method') => void;
  device?: string;
  isOnline?: boolean;
  onPrintReport?: () => void;
  canPrint?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeScreen,
  onScreenChange,
  device = 'MPS',
  isOnline = true,
  onPrintReport,
  canPrint = false,
}) => {
  const navItems = [
    { id: 'analyse', label: 'Analysis Lab', icon: <Activity className="w-4 h-4" /> },
    { id: 'stress-test', label: 'Codec Stress Test', icon: <Sliders className="w-4 h-4" /> },
    { id: 'benchmarks', label: 'Benchmarks & Metrics', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'method', label: 'Methodology & Ethics', icon: <BookOpen className="w-4 h-4" /> },
  ] as const;

  return (
    <header className="border-b border-borderSubtle bg-surface sticky top-0 z-40 select-none shadow-sm">
      <div className="max-w-[1200px] mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
        {/* Left: Brand Identity */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-[8px] bg-primary/10 text-primary border border-primary/20 flex items-center justify-center flex-shrink-0">
            <Shield className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-base tracking-tight text-txt-primary">
                Acoustic Forensics
              </span>
              <span className="text-[11px] font-medium px-2 py-0.5 bg-elevated border border-borderSubtle text-txt-secondary rounded-full">
                v2.4
              </span>
            </div>
            <p className="text-xs text-txt-secondary hidden sm:block">
              Synthetic Speech & Voice Deepfake Screening
            </p>
          </div>
        </div>

        {/* Center: Navigation Pills */}
        <nav className="flex items-center gap-1 p-1 bg-elevated border border-borderSubtle rounded-[10px]">
          {navItems.map((item) => {
            const isActive = activeScreen === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onScreenChange(item.id)}
                className={clsx(
                  'flex items-center gap-2 px-3 py-1.5 rounded-[8px] text-xs font-medium transition-colors duration-150 min-h-[36px]',
                  isActive
                    ? 'bg-surface text-txt-primary shadow-sm border border-borderSubtle font-medium'
                    : 'text-txt-secondary hover:text-txt-primary hover:bg-surface/50 border border-transparent'
                )}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right: Telemetry & Actions */}
        <div className="flex items-center gap-3">
          {/* Hardware & Online Badge */}
          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 bg-elevated border border-borderSubtle rounded-[8px] text-xs text-txt-secondary">
            <span
              className={clsx(
                'w-2 h-2 rounded-full inline-block',
                isOnline ? 'bg-genuine' : 'bg-spoof'
              )}
            />
            <span className="font-medium text-txt-primary">
              {isOnline ? `Online (${device.toUpperCase()})` : 'Offline'}
            </span>
          </div>

          {/* Export Report Action */}
          {canPrint && (
            <Button
              variant="secondary"
              size="sm"
              icon={<Printer className="w-3.5 h-3.5 text-txt-secondary" />}
              onClick={onPrintReport}
              className="hidden md:inline-flex"
            >
              Export Report
            </Button>
          )}
        </div>
      </div>
    </header>
  );
};
