import React from 'react';
import { Cpu, ShieldCheck, Zap } from 'lucide-react';
import { clsx } from 'clsx';
import { ModelInfoItem } from '../../types/api';

export interface ModelSelectorProps {
  models: ModelInfoItem[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  disabled?: boolean;
  className?: string;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  disabled = false,
  className,
}) => {
  const activeModel = models.find((m) => m.id === selectedModelId) || models[0];

  return (
    <div
      className={clsx(
        'p-3.5 rounded-[12px] bg-surface border border-borderSubtle shadow-card flex flex-col md:flex-row md:items-center justify-between gap-3 select-none animate-card-mount',
        className
      )}
    >
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-[8px] bg-primary/10 text-primary border border-primary/20 flex items-center justify-center flex-shrink-0">
          <Cpu className="w-4 h-4" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wider font-semibold text-txt-secondary">
              Active Evaluation Model:
            </span>
            {activeModel?.is_primary && (
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                <ShieldCheck className="w-3 h-3" />
                Production Primary
              </span>
            )}
          </div>
          <p className="text-xs text-txt-secondary mt-0.5 max-w-xl truncate">
            {activeModel?.description || 'Select an architecture for synthetic voice screening.'}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <select
          value={selectedModelId}
          onChange={(e) => onSelectModel(e.target.value)}
          disabled={disabled}
          className="px-3 py-1.5 bg-inset border border-borderSubtle rounded-[8px] text-xs font-medium text-txt-primary focus:outline-none focus:border-primary transition-colors cursor-pointer disabled:opacity-50 min-h-[36px]"
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

