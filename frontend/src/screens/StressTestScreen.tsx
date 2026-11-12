import React, { useState } from 'react';
import { Sliders, CheckCircle2, AlertOctagon, HelpCircle, RefreshCw, Info } from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from '../components/ui/Button';
import { Panel } from '../components/ui/Panel';
import { Readout } from '../components/ui/Readout';
import { AnalysisResult, StressTestResponse } from '../types/api';

export interface StressTestScreenProps {
  activeResult: AnalysisResult | null;
}

export const StressTestScreen: React.FC<StressTestScreenProps> = ({ activeResult }) => {
  const [isTesting, setIsTesting] = useState(false);
  const [stressData, setStressData] = useState<StressTestResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const runStressTest = async () => {
    if (!activeResult) return;
    setIsTesting(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('job_id', activeResult.job_id);

    try {
      const resp = await fetch('/api/stress-test', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: 'Stress test failed' }));
        throw new Error(err.detail || 'Failed to execute codec stress test');
      }

      const data: StressTestResponse = await resp.json();
      setStressData(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Stress test failed');
    } finally {
      setIsTesting(false);
    }
  };

  const getVerdictStyle = (verdict: string) => {
    if (verdict.includes('GENUINE')) {
      return {
        color: 'var(--color-genuine)',
        bg: 'bg-genuine/10',
        border: 'border-genuine/25',
        textColor: 'text-genuine',
        icon: <CheckCircle2 className="w-3.5 h-3.5 text-genuine flex-shrink-0" />,
        label: 'Genuine',
      };
    } else if (verdict.includes('SYNTHETIC')) {
      return {
        color: 'var(--color-spoof)',
        bg: 'bg-spoof/10',
        border: 'border-spoof/25',
        textColor: 'text-spoof',
        icon: <AlertOctagon className="w-3.5 h-3.5 text-spoof flex-shrink-0" />,
        label: 'Synthetic',
      };
    } else {
      return {
        color: 'var(--color-uncertain)',
        bg: 'bg-uncertain/10',
        border: 'border-uncertain/25',
        textColor: 'text-uncertain',
        icon: <HelpCircle className="w-3.5 h-3.5 text-uncertain flex-shrink-0" />,
        label: 'Uncertain',
      };
    }
  };

  return (
    <div className="max-w-[1200px] mx-auto px-4 sm:px-6 py-6 flex flex-col gap-6 select-none animate-cardFadeUp">
      {}
      <div className="border border-borderSubtle bg-surface p-5 sm:p-6 rounded-[12px] shadow-card flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-[8px] bg-primary/10 border border-primary/20 flex items-center justify-center">
              <Sliders className="w-4 h-4 text-primary" />
            </div>
            <h2 className="text-base font-semibold text-txt-primary">
              Audio Codec Compression Stress Testing
            </h2>
          </div>
          <p className="text-xs text-txt-secondary mt-1.5 max-w-2xl leading-relaxed">
            Re-encodes the loaded audio across standard lossy codecs (MP3, AAC, and Opus at 16, 32, and 64 kbps) to verify detection stability against real-world media recompression.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {activeResult ? (
            <Button
              variant="primary"
              size="md"
              icon={<RefreshCw className={clsx('w-4 h-4', isTesting && 'animate-spin')} />}
              onClick={runStressTest}
              disabled={isTesting}
            >
              {isTesting ? 'Transcoding 9 Formats...' : 'Run Stress Test'}
            </Button>
          ) : (
            <span className="text-xs text-txt-muted px-3 py-2 bg-elevated rounded-[8px] border border-borderSubtle">
              Please analyze an audio clip first
            </span>
          )}
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 bg-spoof/10 border border-spoof/25 rounded-[10px] text-xs text-spoof leading-relaxed">
          {errorMsg}
        </div>
      )}

      {}
      {stressData ? (
        <div className="flex flex-col gap-6">
          {}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Readout
              label="Original Score"
              value={stressData.source_score.toFixed(3)}
              sublabel={`Source Verdict: ${stressData.source_verdict}`}
            />
            <Readout
              label="Maximum Score Drift"
              value={Math.max(...stressData.items.map((m) => Math.abs(m.score_shift))).toFixed(3)}
              sublabel="Delta from uncompressed baseline"
              indicatorColor="uncertain"
            />
            <Readout
              label="Verdict Stability"
              value={`${stressData.items.filter((m) => m.verdict === stressData.source_verdict).length} / 9`}
              sublabel="Codecs preserving original verdict"
              indicatorColor={
                stressData.items.every((m) => m.verdict === stressData.source_verdict)
                  ? 'genuine'
                  : 'uncertain'
              }
            />
          </div>

          {}
          <Panel title="Degradation Matrix by Codec & Bitrate">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {(['mp3', 'aac', 'opus'] as const).map((codecName) => {
                const codecItems = stressData.items.filter((m) => m.codec.toLowerCase() === codecName);
                return (
                  <div
                    key={codecName}
                    className="border border-borderSubtle bg-elevated/40 rounded-[10px] p-4 flex flex-col gap-3"
                  >
                    <div className="flex items-center justify-between border-b border-borderSubtle pb-2.5">
                      <span className="text-xs font-semibold uppercase tracking-wider text-txt-primary">
                        {codecName.toUpperCase()} Codec
                      </span>
                      <span className="text-[11px] text-txt-muted">
                        16, 32, 64 kbps
                      </span>
                    </div>

                    <div className="space-y-2">
                      {codecItems.map((item) => {
                        const vStyle = getVerdictStyle(item.verdict);
                        return (
                          <div
                            key={item.bitrate_kbps}
                            className="p-3 bg-surface border border-borderSubtle rounded-[8px] flex flex-col gap-2 hover:border-borderStrong transition-colors duration-150"
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-mono text-xs font-semibold text-txt-primary">
                                {item.bitrate_kbps} kbps
                              </span>
                              <div
                                className={clsx(
                                  'flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-[11px] font-medium',
                                  vStyle.bg,
                                  vStyle.border,
                                  vStyle.textColor
                                )}
                              >
                                {vStyle.icon}
                                <span>{vStyle.label}</span>
                              </div>
                            </div>

                            <div className="flex items-center justify-between text-xs text-txt-secondary">
                              <span>Confidence:</span>
                              <span className="font-mono font-medium text-txt-primary tabular-nums">
                                {(item.probability_bonafide * 100).toFixed(1)}%
                              </span>
                            </div>

                            <div className="flex items-center justify-between text-xs text-txt-secondary">
                              <span>Score Shift:</span>
                              <span
                                className={clsx(
                                  'font-mono tabular-nums font-medium',
                                  item.score_shift < 0 ? 'text-spoof' : 'text-genuine'
                                )}
                              >
                                {item.score_shift > 0 ? `+${item.score_shift.toFixed(3)}` : item.score_shift.toFixed(3)}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </Panel>
        </div>
      ) : (
        <div className="p-10 border border-borderSubtle bg-surface shadow-card rounded-[12px] text-center flex flex-col items-center justify-center gap-3">
          <div className="w-10 h-10 rounded-[8px] bg-elevated border border-borderSubtle flex items-center justify-center text-txt-muted">
            <Info className="w-5 h-5 text-txt-secondary" />
          </div>
          <h3 className="text-sm font-semibold text-txt-primary">
            No Stress Test Run Yet
          </h3>
          <p className="text-xs text-txt-secondary max-w-md leading-relaxed">
            Run the stress testing matrix to evaluate whether compression algorithms like MP3, AAC, and Opus alter synthetic speech detection cues.
          </p>
        </div>
      )}
    </div>
  );
};
