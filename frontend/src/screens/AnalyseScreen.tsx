import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, RotateCcw, AlertCircle, AudioWaveform as WaveformIcon } from 'lucide-react';
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
import { AnalysisResult, PipelineEvent, SampleItem } from '../types/api';

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

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

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
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
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

    const blobUrl = URL.createObjectURL(file);
    setAudioUrl(blobUrl);

    try {
      const formData = new FormData();
      formData.append('audio_file', file);

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

    try {
      const sampleAudioUrl = `/api/samples/${sample.id}/audio`;
      const resp = await fetch(sampleAudioUrl);
      if (!resp.ok) throw new Error('Could not fetch reference audio.');
      const blob = await resp.blob();
      const file = new File([blob], sample.filename, { type: 'audio/flac' });

      handleFileSelected(file);
    } catch (err: any) {
      setErrorMessage(err.message || 'Error loading sample clip.');
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
    } catch (err) {
      console.error('Error fetching result:', err);
    }
  };

  const duration = result?.quality?.duration_sec || 4.0;

  return (
    <div className="max-w-[1200px] mx-auto px-4 sm:px-6 py-6 flex flex-col gap-6">
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
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
        {}
        <div className="md:col-span-5 lg:col-span-4 flex flex-col gap-6">
          <Panel title="Audio Recording">
            <AudioDropzone
              onFileSelected={handleFileSelected}
              onSampleSelected={handleSampleSelected}
              samples={samples}
              isProcessing={isProcessing}
            />
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
                  sublabel="4.0s normalized"
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
                  sublabel="Spectral boundary"
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
        <div className="md:col-span-7 lg:col-span-8 flex flex-col gap-6">
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
                      Forensic Findings
                    </span>
                    <span className="text-xs font-mono text-txt-secondary tabular-nums">
                      Inference: {result.latency_ms.toFixed(0)} ms
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
                Ready for Voice Analysis
              </h3>
              <p className="text-sm text-txt-secondary max-w-md leading-relaxed">
                Upload any spoken voice recording or pick a reference sample from the left panel. The stacked ensemble will evaluate genuine acoustic resonance versus synthetic speech artifacts.
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
            <ModelBreakdownChart
              models={result.models}
              fusedScore={result.fused_score}
              finalVerdict={result.verdict}
            />
          )}
        </div>
      </div>
    </div>
  );
};
