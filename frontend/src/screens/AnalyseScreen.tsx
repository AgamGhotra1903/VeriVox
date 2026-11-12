import React, { useState, useRef, useEffect } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  AlertCircle,
  AudioWaveform as WaveformIcon,
  Upload,
  Bookmark,
  Database,
  Layers,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from '../components/ui/Button';
import { Panel } from '../components/ui/Panel';
import { Readout } from '../components/ui/Readout';
import { Gauge } from '../components/ui/Gauge';
import { AudioDropzone } from '../components/visualizations/AudioDropzone';
import { PipelineTracker } from '../components/visualizations/PipelineTracker';
import { WaveformCanvas } from '../components/visualizations/WaveformCanvas';
import { AttentionHeatstrip } from '../components/visualizations/AttentionHeatstrip';
import { SpectrogramCanvas } from '../components/visualizations/SpectrogramCanvas';
import { ModelBreakdownChart } from '../components/visualizations/ModelBreakdownChart';
import { GroundTruthCard } from '../components/visualizations/GroundTruthCard';
import { DatasetBrowser } from '../components/dataset/DatasetBrowser';
import { ModelSelector } from '../components/layout/ModelSelector';
import {
  AnalysisResult,
  DatasetSampleItem,
  ModelInfoItem,
  PipelineEvent,
  SampleItem,
} from '../types/api';

export interface AnalyseScreenProps {
  samples: SampleItem[];
  onAnalysisComplete?: (result: AnalysisResult) => void;
  activeResult?: AnalysisResult | null;
}

export const AnalyseScreen: React.FC<AnalyseScreenProps> = ({
  samples,
  onAnalysisComplete,
  activeResult: initialResult,
}) => {
  const [result, setResult] = useState<AnalysisResult | null>(initialResult || null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [models, setModels] = useState<ModelInfoItem[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('CRNN-LFCC');

  const [inputTab, setInputTab] = useState<'dataset' | 'upload' | 'reference'>('dataset');
  const [selectedDatasetSampleId, setSelectedDatasetSampleId] = useState<string | null>(null);

  const [sessionTally, setSessionTally] = useState({
    total: 0,
    correct: 0,
    incorrect: 0,
    inconclusive: 0,
  });

  const [showModelComparisonTable, setShowModelComparisonTable] = useState(false);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    fetch('/api/models')
      .then((res) => (res.ok ? res.json() : []))
      .then((data: ModelInfoItem[]) => {
        if (data.length > 0) setModels(data);
      })
      .catch((err) => console.error('Failed to load models list:', err));
  }, []);

  useEffect(() => {
    return () => {
      if (audioUrl && audioUrl.startsWith('blob:')) {
        URL.revokeObjectURL(audioUrl);
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [audioUrl]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName)) {
        e.preventDefault();
        togglePlayPause();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, audioUrl]);

  const togglePlayPause = () => {
    if (!audioRef.current || !audioUrl) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().catch(console.error);
      setIsPlaying(true);
    }
  };

  const handleAudioTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const handleSeek = (time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const handleFileSelected = async (file: File) => {
    setIsProcessing(true);
    setEvents([]);
    setErrorMessage(null);
    setResult(null);
    setSelectedDatasetSampleId(null);

    const blobUrl = URL.createObjectURL(file);
    setAudioUrl(blobUrl);

    try {
      const formData = new FormData();
      formData.append('audio', file);
      formData.append('audio_file', file);
      formData.append('model', selectedModelId);

      const resp = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({ detail: 'Analysis failed.' }));
        throw new Error(errData.detail || 'Failed to process audio.');
      }

      const { job_id } = await resp.json();
      connectSseStream(job_id);
    } catch (err: any) {
      setErrorMessage(err.message || 'Error occurred during audio analysis.');
      setIsProcessing(false);
    }
  };

  const handleSampleSelected = async (sample: SampleItem) => {
    setIsProcessing(true);
    setEvents([]);
    setErrorMessage(null);
    setResult(null);
    setSelectedDatasetSampleId(null);

    try {
      const sampleAudioUrl = `/api/samples/${sample.filename}`;
      const resp = await fetch(sampleAudioUrl);
      if (!resp.ok) throw new Error('Could not fetch reference audio.');
      const blob = await resp.blob();
      const file = new File([blob], sample.filename, { type: 'audio/wav' });
      handleFileSelected(file);
    } catch (err: any) {
      setErrorMessage(err.message || 'Error loading reference clip.');
      setIsProcessing(false);
    }
  };

  const handleDatasetSampleSelected = async (sample: DatasetSampleItem) => {
    setIsProcessing(true);
    setEvents([]);
    setErrorMessage(null);
    setResult(null);
    setSelectedDatasetSampleId(sample.file_id);

    const streamUrl = `/api/dataset/audio/${sample.file_id}`;
    setAudioUrl(streamUrl);

    try {
      const resp = await fetch(
        `/api/dataset/evaluate/${sample.file_id}?model=${encodeURIComponent(selectedModelId)}`,
        { method: 'POST' }
      );
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({ detail: 'Dataset evaluation request failed.' }));
        throw new Error(errData.detail || 'Failed to start evaluation on dataset sample.');
      }

      const { job_id } = await resp.json();
      connectSseStream(job_id);
    } catch (err: any) {
      setErrorMessage(err.message || 'Error evaluating dataset clip.');
      setIsProcessing(false);
    }
  };

  const connectSseStream = (jobId: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = new EventSource(`/api/analyze/${jobId}/events`);
    eventSourceRef.current = es;

    es.addEventListener('pipeline_event', (e) => {
      try {
        const eventData: PipelineEvent = JSON.parse(e.data);
        setEvents((prev) => [...prev, eventData]);

        if (eventData.stage === 'error') {
          setErrorMessage(eventData.error || 'Pipeline execution failed.');
          setIsProcessing(false);
          es.close();
        } else if (eventData.stage === 'done') {
          setIsProcessing(false);
          es.close();
          fetchAnalysisResult(jobId);
        }
      } catch (err) {
        console.error('Failed to parse SSE event:', err);
      }
    });

    es.onerror = () => {
      es.close();
      fetchAnalysisResult(jobId);
    };
  };

  const fetchAnalysisResult = async (jobId: string) => {
    try {
      const resp = await fetch(`/api/result/${jobId}`);
      if (!resp.ok) throw new Error('Result not yet ready');
      const data: AnalysisResult = await resp.json();
      setResult(data);
      if (onAnalysisComplete) {
        onAnalysisComplete(data);
      }

      if (data.ground_truth) {
        setSessionTally((prev) => {
          if (data.is_correct === true) {
            return { ...prev, total: prev.total + 1, correct: prev.correct + 1 };
          } else if (data.is_correct === false) {
            return { ...prev, total: prev.total + 1, incorrect: prev.incorrect + 1 };
          } else {
            return { ...prev, total: prev.total + 1, inconclusive: prev.inconclusive + 1 };
          }
        });
      }
    } catch (err) {
      console.error('Error fetching result:', err);
    }
  };

  const handleResetTally = () => {
    setSessionTally({ total: 0, correct: 0, incorrect: 0, inconclusive: 0 });
  };

  const duration = result?.quality?.duration_sec || 4.0;

  return (
    <div className="max-w-[1240px] mx-auto px-4 sm:px-6 py-6 flex flex-col gap-6">
      {}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={handleAudioTimeUpdate}
          onEnded={handleAudioEnded}
          preload="auto"
        />
      )}

      {}
      {models.length > 0 && (
        <ModelSelector
          models={models}
          selectedModelId={selectedModelId}
          onSelectModel={setSelectedModelId}
          disabled={isProcessing}
        />
      )}

      {}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <Panel
            title="Audio Input Source"
            actions={
              <div className="flex items-center gap-1 p-0.5 bg-inset border border-borderSubtle rounded-[8px]">
                <button
                  onClick={() => setInputTab('dataset')}
                  className={clsx(
                    'px-2.5 py-1 rounded-[6px] text-xs font-medium flex items-center gap-1.5 transition-colors',
                    inputTab === 'dataset'
                      ? 'bg-surface text-txt-primary shadow-sm font-semibold'
                      : 'text-txt-secondary hover:text-txt-primary'
                  )}
                >
                  <Database className="w-3.5 h-3.5 text-primary" />
                  <span>Test Dataset</span>
                </button>
                <button
                  onClick={() => setInputTab('upload')}
                  className={clsx(
                    'px-2.5 py-1 rounded-[6px] text-xs font-medium flex items-center gap-1.5 transition-colors',
                    inputTab === 'upload'
                      ? 'bg-surface text-txt-primary shadow-sm font-semibold'
                      : 'text-txt-secondary hover:text-txt-primary'
                  )}
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload / Mic</span>
                </button>
                <button
                  onClick={() => setInputTab('reference')}
                  className={clsx(
                    'px-2.5 py-1 rounded-[6px] text-xs font-medium flex items-center gap-1.5 transition-colors',
                    inputTab === 'reference'
                      ? 'bg-surface text-txt-primary shadow-sm font-semibold'
                      : 'text-txt-secondary hover:text-txt-primary'
                  )}
                >
                  <Bookmark className="w-3.5 h-3.5" />
                  <span>Reference</span>
                </button>
              </div>
            }
          >
            {inputTab === 'dataset' && (
              <DatasetBrowser
                onSelectSample={handleDatasetSampleSelected}
                selectedSampleId={selectedDatasetSampleId}
                isProcessing={isProcessing}
                sessionTally={sessionTally}
                onResetTally={handleResetTally}
              />
            )}

            {inputTab === 'upload' && (
              <AudioDropzone
                onFileSelected={handleFileSelected}
                onSampleSelected={handleSampleSelected}
                samples={[]}
                isProcessing={isProcessing}
              />
            )}

            {inputTab === 'reference' && (
              <div className="flex flex-col gap-2">
                <p className="text-xs text-txt-secondary mb-1">
                  Curated voice samples with confirmed ground truth for immediate validation:
                </p>
                <div className="grid grid-cols-1 gap-2">
                  {samples.map((sample) => (
                    <button
                      key={sample.id}
                      onClick={() => handleSampleSelected(sample)}
                      disabled={isProcessing}
                      className="flex items-center justify-between p-3 rounded-[10px] bg-surface hover:bg-elevated border border-borderSubtle text-left transition-colors duration-150 group disabled:opacity-50 min-h-[44px]"
                    >
                      <div className="flex items-center gap-2.5 min-w-0">
                        <div className="w-7 h-7 rounded-[6px] bg-elevated border border-borderSubtle flex items-center justify-center flex-shrink-0 text-txt-secondary group-hover:text-primary transition-colors">
                          <Play className="w-3.5 h-3.5" />
                        </div>
                        <div className="truncate">
                          <p className="text-xs font-medium text-txt-primary truncate group-hover:text-primary transition-colors">
                            {sample.name}
                          </p>
                          <p className="text-[11px] text-txt-secondary truncate">
                            {sample.description}
                          </p>
                        </div>
                      </div>

                      <span
                        className={clsx(
                          'text-[11px] font-medium px-2 py-0.5 rounded-full border flex-shrink-0 ml-2',
                          sample.ground_truth === 'bonafide'
                            ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                            : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                        )}
                      >
                        {sample.ground_truth === 'bonafide' ? 'Genuine' : 'Deepfake'}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </Panel>

          {}
          {(isProcessing || events.length > 0) && (
            <PipelineTracker
              events={events}
              isComplete={!isProcessing && result !== null}
              error={errorMessage}
            />
          )}

          {}
          {result && (
            <Panel title="Acoustic Quality Profile">
              <div className="grid grid-cols-2 gap-3">
                <Readout
                  label="Duration"
                  value={result.quality.duration_sec.toFixed(2)}
                  unit="sec"
                  sublabel="4.0s normalized window"
                />
                <Readout
                  label="Sample Rate"
                  value={`${result.quality.sample_rate / 1000}`}
                  unit="kHz"
                  sublabel="Mono uncompressed"
                />
                <Readout
                  label="SNR Estimate"
                  value={result.quality.snr_db.toFixed(1)}
                  unit="dB"
                  indicatorColor={result.quality.snr_db > 20 ? 'genuine' : 'uncertain'}
                  sublabel={result.quality.snr_db > 20 ? 'High clarity' : 'Moderate noise'}
                />
                <Readout
                  label="Clipping"
                  value={(result.quality.clipping_ratio * 100).toFixed(2)}
                  unit="%"
                  indicatorColor={result.quality.clipping_ratio === 0 ? 'genuine' : 'spoof'}
                  sublabel={result.quality.clipping_ratio === 0 ? 'No saturation' : 'Peak clipped'}
                />
                <Readout
                  label="Bandwidth"
                  value={Math.round(result.quality.bandwidth_hz)}
                  unit="Hz"
                  sublabel={result.quality.bandwidth_hz >= 6000 ? 'Wideband Nyquist' : 'Band-limited'}
                />
                <Readout
                  label="Vocal Energy"
                  value={((1 - result.quality.silence_ratio) * 100).toFixed(0)}
                  unit="%"
                  sublabel="Active speech density"
                />
              </div>
            </Panel>
          )}
        </div>

        {}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {}
          {result && result.ground_truth && (
            <GroundTruthCard
              groundTruth={result.ground_truth}
              attackId={result.attack_id}
              verdict={result.verdict}
              isCorrect={result.is_correct}
            />
          )}

          {}
          {result ? (
            <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch animate-card-mount">
              {}
              <div className="md:col-span-5 flex">
                <Gauge
                  probability={result.probability_bonafide}
                  verdict={result.verdict}
                  threshold={result.decision_threshold}
                  className="w-full"
                />
              </div>

              {}
              <div className="md:col-span-7 flex flex-col justify-between p-5 sm:p-6 bg-surface border border-borderSubtle rounded-[12px] shadow-card">
                <div>
                  <div className="flex items-center justify-between pb-3 border-b border-borderSubtle mb-3">
                    <span className="text-xs uppercase tracking-wide font-semibold text-txt-secondary">
                      Forensic Findings ({result.primary_system || 'Active Model'})
                    </span>
                    <span className="text-xs font-mono text-txt-secondary tabular-nums">
                      Latency: {result.latency_ms.toFixed(0)} ms
                    </span>
                  </div>

                  <p className="text-sm text-txt-primary leading-relaxed">
                    {result.explanation}
                  </p>
                </div>

                {}
                <div className="mt-4 p-3 bg-amber-50 border border-amber-200 text-amber-900 dark:bg-amber-950/20 dark:border-amber-900/40 dark:text-amber-200 rounded-[8px] text-xs flex items-start gap-2.5">
                  <AlertCircle className="w-4 h-4 text-uncertain flex-shrink-0 mt-0.5" />
                  <span className="leading-snug">{result.caveat}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-8 sm:p-12 rounded-[12px] border border-borderSubtle bg-surface text-center flex flex-col items-center justify-center gap-3 shadow-card animate-card-mount">
              <div className="w-12 h-12 rounded-[10px] bg-primary/10 text-primary border border-primary/20 flex items-center justify-center">
                <WaveformIcon className="w-6 h-6" />
              </div>
              <h3 className="text-base font-semibold text-txt-primary">
                Ready for Voice Screening
              </h3>
              <p className="text-sm text-txt-secondary max-w-md leading-relaxed">
                Select an audio clip from our ASVspoof 2019 test corpus, pick a reference sample, or upload your own recording to begin real-time acoustic forensic evaluation.
              </p>
            </div>
          )}

          {}
          {result && (
            <Panel
              title="Time-Frequency Signal Inspection"
              actions={
                <div className="flex items-center gap-2">
                  <Button
                    variant="primary"
                    size="sm"
                    icon={isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                    onClick={togglePlayPause}
                  >
                    {isPlaying ? 'Pause' : 'Play Audio'}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    icon={<RotateCcw className="w-3.5 h-3.5" />}
                    onClick={() => handleSeek(0)}
                    title="Rewind to start"
                  >
                    Reset
                  </Button>
                  <span className="text-xs font-mono text-txt-secondary ml-1 tabular-nums">
                    {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
                  </span>
                </div>
              }
            >
              <div className="flex flex-col gap-4">
                {}
                <div>
                  <div className="flex items-center justify-between text-xs text-txt-secondary mb-1.5">
                    <span className="font-medium text-txt-primary">Waveform Amplitude (16 kHz Peak Envelope)</span>
                    <span className="text-xs text-txt-muted">Click or drag along waveform to scrub</span>
                  </div>
                  <WaveformCanvas
                    peaks={result.waveform_peaks}
                    durationSec={duration}
                    currentTime={currentTime}
                    onSeek={handleSeek}
                    height={90}
                  />
                </div>

                {}
                <AttentionHeatstrip
                  attentionWeights={result.attention_weights}
                  durationSec={duration}
                  currentTime={currentTime}
                  height={18}
                />

                {}
                <SpectrogramCanvas
                  lfccData={result.lfcc_downsampled}
                  logmelData={result.logmel_downsampled}
                  durationSec={duration}
                  currentTime={currentTime}
                  height={120}
                />
              </div>
            </Panel>
          )}

          {}
          {result && (
            <div className="flex flex-col gap-3">
              <ModelBreakdownChart
                models={result.models}
                fusedScore={result.fused_score}
                finalVerdict={result.verdict}
              />

              <div className="p-3.5 rounded-[10px] bg-surface border border-borderSubtle flex flex-col gap-3">
                <button
                  onClick={() => setShowModelComparisonTable((prev) => !prev)}
                  className="flex items-center justify-between text-xs font-semibold text-txt-primary hover:text-primary transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-primary" />
                    <span>Multi-Model Decision Comparison Matrix ({result.models.length} Systems)</span>
                  </div>
                  {showModelComparisonTable ? (
                    <ChevronUp className="w-4 h-4 text-txt-secondary" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-txt-secondary" />
                  )}
                </button>

                {showModelComparisonTable && (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs font-mono">
                      <thead>
                        <tr className="border-b border-borderSubtle text-[11px] text-txt-muted uppercase">
                          <th className="py-2 px-3">System Name</th>
                          <th className="py-2 px-3">Raw Continuous Score</th>
                          <th className="py-2 px-3">Stacker Weight</th>
                          <th className="py-2 px-3">System Verdict</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-borderSubtle">
                        {result.models.map((m) => (
                          <tr key={m.name} className="hover:bg-elevated/40 transition-colors">
                            <td className="py-2 px-3 font-sans font-medium text-txt-primary">
                              {m.name}
                            </td>
                            <td className="py-2 px-3 text-txt-primary tabular-nums">
                              {m.score > 0 ? `+${m.score.toFixed(4)}` : m.score.toFixed(4)}
                            </td>
                            <td className="py-2 px-3 text-txt-secondary tabular-nums">
                              {m.ensemble_weight.toFixed(4)}
                            </td>
                            <td className="py-2 px-3">
                              <span
                                className={clsx(
                                  'px-2 py-0.5 rounded-full text-[10px] font-sans font-semibold border',
                                  m.verdict === 'BONA FIDE (GENUINE)'
                                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                                    : m.verdict === 'SPOOF (SYNTHETIC)'
                                    ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                                    : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                                )}
                              >
                                {m.verdict}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
