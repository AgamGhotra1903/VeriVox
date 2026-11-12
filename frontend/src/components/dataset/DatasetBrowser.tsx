import React, { useState, useEffect, useCallback } from 'react';
import { Search, Shuffle, Play, ChevronLeft, ChevronRight, RotateCcw, Shield, Bug, Sparkles } from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from '../ui/Button';
import { DatasetSampleItem, DatasetSummaryResponse } from '../../types/api';

export interface DatasetBrowserProps {
  onSelectSample: (sample: DatasetSampleItem) => void;
  selectedSampleId?: string | null;
  isProcessing: boolean;
  sessionTally: {
    total: number;
    correct: number;
    incorrect: number;
    inconclusive: number;
  };
  onResetTally: () => void;
  className?: string;
}

const ATTACK_OPTIONS = [
  { id: 'all', label: 'All Attacks (A07 - A19)' },
  { id: 'A07', label: 'A07: Vocoder WORLD' },
  { id: 'A08', label: 'A08: Neural Source-Filter' },
  { id: 'A09', label: 'A09: Vocoder STRAIGHT' },
  { id: 'A10', label: 'A10: Neural TTS Tacotron 2' },
  { id: 'A11', label: 'A11: TTS Griffin-Lim' },
  { id: 'A12', label: 'A12: Neural WaveNet VC' },
  { id: 'A13', label: 'A13: CycleGAN VC' },
  { id: 'A14', label: 'A14: VAE-based VC' },
  { id: 'A15', label: 'A15: StarGAN VC' },
  { id: 'A16', label: 'A16: Spectral Filtering VC' },
  { id: 'A17', label: 'A17: WaveNet Vocoder' },
  { id: 'A18', label: 'A18: Mel Inversion' },
  { id: 'A19', label: 'A19: Concatenation' },
];

export const DatasetBrowser: React.FC<DatasetBrowserProps> = ({
  onSelectSample,
  selectedSampleId,
  isProcessing,
  sessionTally,
  onResetTally,
  className,
}) => {
  const [summary, setSummary] = useState<DatasetSummaryResponse | null>(null);
  const [samples, setSamples] = useState<DatasetSampleItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const pageSize = 8;
  const [labelFilter, setLabelFilter] = useState<'all' | 'bonafide' | 'spoof'>('all');
  const [attackFilter, setAttackFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isPickingRandom, setIsPickingRandom] = useState(false);

  useEffect(() => {
    fetch('/api/dataset/summary')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setSummary(data);
      })
      .catch((err) => console.error('Failed to fetch dataset summary:', err));
  }, []);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedSearch(searchQuery);
      setPage(1);
    }, 300);
    return () => clearTimeout(handler);
  }, [searchQuery]);

  const loadSamples = useCallback(async () => {
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        page: page.toString(),
        page_size: pageSize.toString(),
      });
      if (labelFilter !== 'all') params.set('label', labelFilter);
      if (attackFilter !== 'all') params.set('attack_id', attackFilter);
      if (debouncedSearch.trim()) params.set('search', debouncedSearch.trim());

      const res = await fetch(`/api/dataset/samples?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data.samples || []);
        setTotal(data.total || 0);
      }
    } catch (err) {
      console.error('Failed to load dataset samples:', err);
    } finally {
      setIsLoading(false);
    }
  }, [page, pageSize, labelFilter, attackFilter, debouncedSearch]);

  useEffect(() => {
    loadSamples();
  }, [loadSamples]);

  const handlePickRandom = async () => {
    if (isProcessing || isPickingRandom) return;
    setIsPickingRandom(true);
    try {
      const params = new URLSearchParams({ random_sample: 'true' });
      if (labelFilter !== 'all') params.set('label', labelFilter);
      if (attackFilter !== 'all') params.set('attack_id', attackFilter);

      const res = await fetch(`/api/dataset/samples?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        if (data.samples && data.samples.length > 0) {
          onSelectSample(data.samples[0]);
        }
      }
    } catch (err) {
      console.error('Failed to pick random sample:', err);
    } finally {
      setIsPickingRandom(false);
    }
  };

  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const accuracyPct =
    sessionTally.total > 0
      ? ((sessionTally.correct / sessionTally.total) * 100).toFixed(1)
      : '0.0';

  return (
    <div className={clsx('flex flex-col gap-3', className)}>
      {}
      <div className="flex items-center justify-between p-2.5 rounded-[10px] bg-elevated/60 border border-borderSubtle text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-txt-primary">Session Scorecard:</span>
          <span className="font-mono text-txt-secondary tabular-nums">
            {sessionTally.total} Tested · {sessionTally.correct} Correct ({accuracyPct}%)
          </span>
          {sessionTally.inconclusive > 0 && (
            <span className="text-amber-400 font-mono text-[11px]">
              ({sessionTally.inconclusive} abstained)
            </span>
          )}
        </div>
        {sessionTally.total > 0 && (
          <button
            onClick={onResetTally}
            className="text-[11px] text-txt-muted hover:text-txt-primary flex items-center gap-1 transition-colors"
            title="Reset session evaluation tally"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Reset</span>
          </button>
        )}
      </div>

      {}
      <div className="flex items-center gap-2">
        <Button
          variant="primary"
          size="sm"
          icon={<Shuffle className="w-3.5 h-3.5" />}
          onClick={handlePickRandom}
          disabled={isProcessing || isPickingRandom}
          className="w-full text-xs font-semibold justify-center py-2"
        >
          {isPickingRandom ? 'Selecting Sample...' : 'Pick Random Test Sample'}
        </Button>
      </div>

      {}
      <div className="flex flex-col gap-2 p-2.5 rounded-[10px] bg-surface border border-borderSubtle">
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-txt-muted absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            placeholder="Search File ID (e.g. LA_E_2833634)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 bg-inset border border-borderSubtle rounded-[8px] text-xs text-txt-primary placeholder-txt-muted focus:outline-none focus:border-primary font-mono transition-colors"
          />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <select
              value={labelFilter}
              onChange={(e) => {
                setLabelFilter(e.target.value as any);
                setPage(1);
              }}
              className="w-full px-2 py-1.5 bg-inset border border-borderSubtle rounded-[8px] text-xs text-txt-primary focus:outline-none focus:border-primary transition-colors cursor-pointer"
            >
              <option value="all">All Labels</option>
              <option value="bonafide">Bona Fide Only</option>
              <option value="spoof">Spoof Only</option>
            </select>
          </div>

          <div>
            <select
              value={attackFilter}
              onChange={(e) => {
                setAttackFilter(e.target.value);
                setPage(1);
              }}
              disabled={labelFilter === 'bonafide'}
              className="w-full px-2 py-1.5 bg-inset border border-borderSubtle rounded-[8px] text-xs text-txt-primary focus:outline-none focus:border-primary transition-colors cursor-pointer disabled:opacity-50"
            >
              {ATTACK_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {}
      <div className="flex items-center justify-between text-[11px] text-txt-muted px-1">
        <span>
          {isLoading ? 'Searching...' : `Found ${total.toLocaleString()} clips`}
        </span>
        <span>
          Page {page} of {totalPages}
        </span>
      </div>

      {}
      <div className="flex flex-col gap-1.5 max-h-[300px] overflow-y-auto pr-1">
        {samples.map((sample) => {
          const isSelected = selectedSampleId === sample.file_id;
          const isBonaFide = sample.label === 'bonafide';

          return (
            <div
              key={sample.file_id}
              onClick={() => !isProcessing && onSelectSample(sample)}
              className={clsx(
                'flex items-center justify-between p-2 rounded-[8px] border text-xs cursor-pointer transition-colors duration-150 select-none group',
                isSelected
                  ? 'bg-primary/10 border-primary/40 text-txt-primary shadow-sm'
                  : 'bg-surface hover:bg-elevated/70 border-borderSubtle text-txt-secondary hover:text-txt-primary'
              )}
            >
              <div className="flex items-center gap-2 min-w-0">
                <div
                  className={clsx(
                    'w-6 h-6 rounded-[6px] border flex items-center justify-center flex-shrink-0 transition-colors',
                    isSelected
                      ? 'bg-primary text-white border-primary'
                      : 'bg-elevated border-borderSubtle text-txt-muted group-hover:text-primary'
                  )}
                >
                  <Play className="w-3 h-3" />
                </div>

                <div className="truncate">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono font-medium text-txt-primary text-xs truncate">
                      {sample.file_id}
                    </span>
                    <span
                      className={clsx(
                        'text-[10px] font-semibold px-1.5 py-0.2 rounded-full border',
                        isBonaFide
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                      )}
                    >
                      {isBonaFide ? 'Genuine' : `Spoof (${sample.attack_id})`}
                    </span>
                  </div>
                  <span className="text-[10px] text-txt-muted">
                    Speaker: {sample.speaker}
                  </span>
                </div>
              </div>

              <Button
                variant={isSelected ? 'primary' : 'secondary'}
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectSample(sample);
                }}
                disabled={isProcessing}
                className="text-[11px] py-1 px-2.5 h-7 flex-shrink-0"
              >
                Analyze
              </Button>
            </div>
          );
        })}

        {samples.length === 0 && !isLoading && (
          <div className="p-6 text-center text-xs text-txt-muted rounded-[8px] bg-elevated/30 border border-borderSubtle">
            No evaluation samples match current filter criteria.
          </div>
        )}
      </div>

      {}
      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-2 border-t border-borderSubtle">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1 || isLoading}
            icon={<ChevronLeft className="w-3.5 h-3.5" />}
            className="text-xs py-1"
          >
            Prev
          </Button>

          <span className="text-xs font-mono text-txt-secondary tabular-nums">
            {page} / {totalPages}
          </span>

          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages || isLoading}
            icon={<ChevronRight className="w-3.5 h-3.5" />}
            className="text-xs py-1"
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
};
