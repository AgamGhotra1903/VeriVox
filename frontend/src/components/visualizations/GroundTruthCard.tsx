import React from 'react';
import { CheckCircle2, AlertOctagon, HelpCircle, Shield, Bug } from 'lucide-react';
import { clsx } from 'clsx';

export interface GroundTruthCardProps {
  groundTruth: 'bonafide' | 'spoof' | string;
  attackId?: string | null;
  verdict: 'BONA FIDE (GENUINE)' | 'SPOOF (SYNTHETIC)' | 'INCONCLUSIVE' | string;
  isCorrect?: boolean | null;
  className?: string;
}

const ATTACK_DESCRIPTIONS: Record<string, string> = {
  '-': 'Authentic Human Speech (Controlled Environment)',
  'A07': 'Vocoder: WORLD (Speech Synthesis / TTS)',
  'A08': 'Vocoder: Neural Source-Filter (NSF)',
  'A09': 'Vocoder: STRAIGHT',
  'A10': 'Neural TTS: Tacotron 2 + WaveRNN',
  'A11': 'Neural TTS: Tacotron 2 + Griffin-Lim',
  'A12': 'Voice Conversion: Neural WaveNet VC',
  'A13': 'Voice Conversion: CycleGAN VC',
  'A14': 'Voice Conversion: VAE-based VC',
  'A15': 'Voice Conversion: StarGAN VC',
  'A16': 'Voice Conversion: Spectral Filtering VC',
  'A17': 'WaveNet Vocoder (Acoustic Model Inversion)',
  'A18': 'Neural Vocoder (Mel-Spectrogram Inversion)',
  'A19': 'Waveform Concatenation Synthesis',
};

export const GroundTruthCard: React.FC<GroundTruthCardProps> = ({
  groundTruth,
  attackId = '-',
  verdict,
  isCorrect,
  className,
}) => {
  const isBonaFide = groundTruth.toLowerCase() === 'bonafide';
  const cleanAttackId = attackId && attackId !== 'None' ? attackId : (isBonaFide ? '-' : 'Unknown');
  const attackDescription = ATTACK_DESCRIPTIONS[cleanAttackId] || `Synthetic Attack Algorithm (${cleanAttackId})`;

  let statusConfig;
  if (isCorrect === true) {
    statusConfig = {
      label: 'Ground Truth Match',
      tag: 'CORRECT DECISION',
      bgColor: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
      badgeBg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
      icon: <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />,
      subtext: 'Model prediction correctly identified true acoustic origin.',
    };
  } else if (isCorrect === false) {
    statusConfig = {
      label: 'Ground Truth Discrepancy',
      tag: 'MISCLASSIFICATION',
      bgColor: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
      badgeBg: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
      icon: <AlertOctagon className="w-4 h-4 text-rose-400 flex-shrink-0" />,
      subtext: 'Model verdict contradicted reference ground truth.',
    };
  } else {
    statusConfig = {
      label: 'Abstention Guard',
      tag: 'INCONCLUSIVE / ABSTAINED',
      bgColor: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
      badgeBg: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
      icon: <HelpCircle className="w-4 h-4 text-amber-400 flex-shrink-0" />,
      subtext: 'Score fell into calibrated uncertainty interval [40% - 60%]. Model safely abstained from false alarm.',
    };
  }

  return (
    <div
      className={clsx(
        'p-4 rounded-[12px] border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 transition-colors duration-150',
        statusConfig.bgColor,
        className
      )}
    >
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-[8px] bg-elevated border border-borderSubtle flex items-center justify-center flex-shrink-0 mt-0.5">
          {isBonaFide ? (
            <Shield className="w-4 h-4 text-emerald-400" />
          ) : (
            <Bug className="w-4 h-4 text-rose-400" />
          )}
        </div>
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-semibold uppercase tracking-wider text-txt-primary">
              Ground Truth Reference:
            </span>
            <span
              className={clsx(
                'text-xs font-semibold px-2 py-0.5 rounded-full border',
                isBonaFide
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                  : 'bg-rose-500/20 text-rose-300 border-rose-500/30'
              )}
            >
              {isBonaFide ? 'Authentic Bona Fide Human Voice' : `Spoof Deepfake (${cleanAttackId})`}
            </span>
          </div>
          <p className="text-xs text-txt-secondary mt-1 leading-snug">
            {attackDescription}
          </p>
        </div>
      </div>

      <div className="flex flex-col sm:items-end gap-1 flex-shrink-0">
        <div className={clsx('px-3 py-1 rounded-full border text-xs font-semibold flex items-center gap-1.5', statusConfig.badgeBg)}>
          {statusConfig.icon}
          <span>{statusConfig.tag}</span>
        </div>
        <span className="text-[11px] text-txt-muted text-right">
          {statusConfig.subtext}
        </span>
      </div>
    </div>
  );
};
