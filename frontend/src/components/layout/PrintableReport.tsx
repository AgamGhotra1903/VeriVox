import React from 'react';
import { AnalysisResult } from '../../types/api';

export interface PrintableReportProps {
  result: AnalysisResult;
}

export const PrintableReport: React.FC<PrintableReportProps> = ({ result }) => {
  return (
    <div className="hidden print:block p-8 bg-white text-black font-mono text-xs max-w-4xl mx-auto">
      {}
      <div className="border-b-2 border-black pb-4 mb-6 flex justify-between items-start">
        <div>
          <h1 className="text-xl font-bold uppercase tracking-wider text-black">
            Acoustic Forensics Laboratory Report
          </h1>
          <p className="text-[10px] text-gray-600 mt-1 uppercase">
            Synthetic Speech & Deepfake Audio Forensic Examination
          </p>
        </div>
        <div className="text-right text-[10px] text-gray-700">
          <div>JOB ID: {result.job_id}</div>
          <div>DATE: {new Date(result.created_at).toUTCString()}</div>
          <div>PROTOCOL: ASVSPOOF 2019 LA</div>
        </div>
      </div>

      {}
      <div className="border-2 border-black p-4 mb-6 bg-gray-50 flex items-center justify-between">
        <div>
          <div className="text-[10px] uppercase text-gray-600 font-bold">
            FORENSIC SCREENER VERDICT
          </div>
          <div className="text-2xl font-bold uppercase tracking-tight mt-1">
            {result.verdict}
          </div>
          <p className="text-[11px] text-gray-700 mt-1 max-w-xl">
            {result.explanation}
          </p>
        </div>
        <div className="text-right border-l-2 border-black pl-6">
          <div className="text-[10px] uppercase text-gray-600 font-bold">P(GENUINE)</div>
          <div className="text-3xl font-bold tracking-tight">
            {(result.probability_bonafide * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-gray-600 mt-0.5">
            Dev Thresh: {result.decision_threshold.toFixed(4)}
          </div>
        </div>
      </div>

      {}
      <div className="mb-6">
        <h2 className="text-xs font-bold uppercase tracking-wider border-b border-black pb-1 mb-2">
          Signal Conditioning Diagnostics
        </h2>
        <div className="grid grid-cols-4 gap-2 text-[10px]">
          <div className="border border-gray-400 p-2">
            <span className="text-gray-600 block">DURATION</span>
            <strong className="text-xs">{result.quality.duration_sec.toFixed(2)}s</strong>
          </div>
          <div className="border border-gray-400 p-2">
            <span className="text-gray-600 block">SAMPLE RATE</span>
            <strong className="text-xs">{result.quality.sample_rate} Hz</strong>
          </div>
          <div className="border border-gray-400 p-2">
            <span className="text-gray-600 block">ESTIMATED SNR</span>
            <strong className="text-xs">{result.quality.snr_db.toFixed(1)} dB</strong>
          </div>
          <div className="border border-gray-400 p-2">
            <span className="text-gray-600 block">BANDWIDTH</span>
            <strong className="text-xs">{Math.round(result.quality.bandwidth_hz)} Hz</strong>
          </div>
        </div>
      </div>

      {}
      <div className="mb-6">
        <h2 className="text-xs font-bold uppercase tracking-wider border-b border-black pb-1 mb-2">
          Ensemble Model Breakdown
        </h2>
        <table className="w-full text-left text-[10px] border-collapse">
          <thead>
            <tr className="border-b border-black text-gray-700">
              <th className="py-1">Model System</th>
              <th className="py-1 text-right">Score</th>
              <th className="py-1 text-right">Ensemble Weight</th>
              <th className="py-1 text-right">Contribution</th>
              <th className="py-1 text-right">Individual Decision</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-300">
            {result.models.map((m) => (
              <tr key={m.name}>
                <td className="py-1 font-semibold">{m.name}</td>
                <td className="py-1 text-right tabular-nums">
                  {m.score > 0 ? `+${m.score.toFixed(3)}` : m.score.toFixed(3)}
                </td>
                <td className="py-1 text-right tabular-nums">{m.ensemble_weight.toFixed(4)}</td>
                <td className="py-1 text-right tabular-nums font-semibold">
                  {m.contribution.toFixed(4)}
                </td>
                <td className="py-1 text-right uppercase">{m.verdict}</td>
              </tr>
            ))}
            <tr className="border-t-2 border-black font-bold">
              <td className="py-1.5">STACKED META-SCORE</td>
              <td className="py-1.5 text-right tabular-nums">
                {result.fused_score > 0 ? `+${result.fused_score.toFixed(4)}` : result.fused_score.toFixed(4)}
              </td>
              <td className="py-1.5 text-right">L2-REG</td>
              <td className="py-1.5 text-right">P={ (result.probability_bonafide*100).toFixed(1) }%</td>
              <td className="py-1.5 text-right uppercase">{result.verdict}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {}
      <div className="border-t border-black pt-3 text-[9px] text-gray-700 leading-relaxed">
        <p className="mb-4">
          <strong>Caveat: </strong>
          {result.caveat}
        </p>

        <div className="flex justify-between items-end pt-6">
          <div>
            <div>ANALYSIS SYSTEM: Acoustic Forensics v0.1.0 (CRNN + GMM + Stacking)</div>
            <div>VERIFIED ZERO-LEAKAGE EVALUATION PIPELINE</div>
          </div>
          <div className="border-t border-black w-48 text-center pt-1 text-[10px]">
            EXAMINER SIGNATURE
          </div>
        </div>
      </div>
    </div>
  );
};

