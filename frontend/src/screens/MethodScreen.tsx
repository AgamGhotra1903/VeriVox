import React from 'react';
import { Shield, Lock, AlertTriangle, Activity, CheckCircle2, BookOpen } from 'lucide-react';
import { Panel } from '../components/ui/Panel';

export const MethodScreen: React.FC = () => {
  const pipelineSteps = [
    {
      num: '01',
      title: 'Signal Conditioning',
      desc: 'Enforces 16,000 Hz single-channel mono, peak normalization (no clipping), and center-crop or repeat-pad windowing to 4.0 seconds (64,000 samples).',
      tag: '16 kHz Audio',
    },
    {
      num: '02',
      title: 'Dual Spectral Extraction',
      desc: 'Computes 60-dim LFCCs (capturing sub-band spectral phase discontinuities) and 80-dim Log-Mel filterbanks. Features normalized via per-utterance CMVN.',
      tag: 'LFCC & Log-Mel',
    },
    {
      num: '03',
      title: '6-Model Base Inference',
      desc: 'Executes CRNN-LFCC (with self-attention), CRNN-LogMel, 64-component diagonal GMM, RBF-SVM, Random Forest, and Logistic Regression on Train-fitted PCA.',
      tag: 'Continuous Scores',
    },
    {
      num: '04',
      title: 'Score Stacking Fusion',
      desc: 'Transforms 6 continuous scores via a Dev-fitted standardizer, passing into an L2-regularized Logistic Regression meta-learner with optimal C hyperparameter.',
      tag: 'L2 Meta-Learner',
    },
    {
      num: '05',
      title: 'Calibrated Verdict',
      desc: 'Computes sigmoid probability P(Bona Fide) with a calibrated uncertainty band [40% - 60%], aligned temporal attention salience, and plain-language evidence note.',
      tag: 'Final Verdict',
    },
  ];

  return (
    <div className="max-w-[1200px] mx-auto px-4 sm:px-6 py-6 flex flex-col gap-6 select-none font-sans animate-cardFadeUp">
      {/* Header Banner */}
      <div className="border border-borderSubtle bg-surface p-5 sm:p-6 rounded-[12px] shadow-card">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-[8px] bg-primary/10 border border-primary/20 flex items-center justify-center">
            <BookOpen className="w-4 h-4 text-primary" />
          </div>
          <h2 className="text-base font-semibold text-txt-primary">
            Acoustic Forensics Methodology & Scientific Rigor
          </h2>
        </div>
        <p className="text-xs text-txt-secondary mt-1.5 leading-relaxed">
          Technical architecture, zero-leakage experimental protocols, and forensic limitations of the 
          6-model stacked ensemble system.
        </p>
      </div>

      {/* Signal Flow Diagram */}
      <Panel title="Forensic Pipeline Signal Flow (End-to-End)">
        <div className="flex flex-col gap-3 py-1">
          {pipelineSteps.map((step) => (
            <div
              key={step.num}
              className="border border-borderSubtle bg-surface p-4 rounded-[10px] flex items-start gap-3.5 hover:border-borderStrong transition-colors duration-150"
            >
              <span className="w-7 h-7 rounded-[8px] bg-primary/10 border border-primary/20 flex items-center justify-center font-semibold text-primary text-xs flex-shrink-0 font-mono">
                {step.num}
              </span>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <h3 className="text-xs font-semibold text-txt-primary">
                    {step.title}
                  </h3>
                  <span className="text-[11px] px-2 py-0.5 bg-elevated border border-borderSubtle text-txt-secondary rounded-full font-medium">
                    {step.tag}
                  </span>
                </div>
                <p className="text-xs text-txt-secondary leading-relaxed">
                  {step.desc}
                </p>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* Zero Data Leakage Rules */}
      <Panel title="Data Leakage Prevention Guarantees">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-surface border border-borderSubtle rounded-[10px] space-y-1.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-txt-primary">
              <Lock className="w-4 h-4 text-primary" />
              <span>Training Partition Isolation</span>
            </div>
            <p className="text-xs text-txt-secondary leading-relaxed">
              Base models (CRNN, GMM, SVM, RF, LR) and feature transformers (PCA, scaler) are fitted 
              <strong> strictly on the official ASVspoof 2019 LA TRAIN partition</strong>. Zero dev or eval data 
              informs base model parameter updates.
            </p>
          </div>

          <div className="p-4 bg-surface border border-borderSubtle rounded-[10px] space-y-1.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-txt-primary">
              <Shield className="w-4 h-4 text-primary" />
              <span>Meta-Learner Calibration Isolation</span>
            </div>
            <p className="text-xs text-txt-secondary leading-relaxed">
              The L2 Logistic Regression meta-learner and score standardizer are fitted 
              <strong> strictly on DEV partition scores</strong>. The EVAL partition remains completely 
              unseen until final one-pass benchmark computation.
            </p>
          </div>

          <div className="p-4 bg-surface border border-borderSubtle rounded-[10px] space-y-1.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-txt-primary">
              <CheckCircle2 className="w-4 h-4 text-primary" />
              <span>Speaker & Audio Hash Audit</span>
            </div>
            <p className="text-xs text-txt-secondary leading-relaxed">
              Automated hash and speaker audits verify:
              (1) zero speaker overlap between train, dev, and eval,
              (2) zero audio content SHA-256 collision, and
              (3) zero utterance ID leakage.
            </p>
          </div>

          <div className="p-4 bg-surface border border-borderSubtle rounded-[10px] space-y-1.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-txt-primary">
              <Activity className="w-4 h-4 text-primary" />
              <span>Sign Convention Synchronization</span>
            </div>
            <p className="text-xs text-txt-secondary leading-relaxed">
              All 6 models are guaranteed to follow the exact same continuous score convention: 
              <strong> higher values indicate authentic bona fide speech</strong>, while negative values 
              indicate synthetic deepfake audio.
            </p>
          </div>
        </div>
      </Panel>

      {/* Honest Scientific Limitations */}
      <div className="p-4 sm:p-5 bg-uncertain/10 border border-uncertain/25 rounded-[12px] flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-uncertain flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="text-xs font-semibold text-uncertain">
            Forensic Scope and Practical Limitations
          </h4>
          <p className="text-xs text-txt-secondary leading-relaxed">
            This system evaluates acoustic and temporal cues characteristic of text-to-speech (TTS) synthesis 
            and voice conversion (VC). Highly compressed lossy audio (e.g. WhatsApp, phone call codecs at 16 kbps) 
            or aggressive studio re-recording may attenuate spectral phase cues. In consequential contexts, decisions 
            should always be corroborated with biological speaker verification and chain-of-custody metadata.
          </p>
        </div>
      </div>
    </div>
  );
};
