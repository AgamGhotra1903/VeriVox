import React, { useRef, useState, useEffect } from 'react';
import { Upload, Mic, MicOff, Play } from 'lucide-react';
import { clsx } from 'clsx';
import { Button } from '../ui/Button';
import { SampleItem } from '../../types/api';

export interface AudioDropzoneProps {
  onFileSelected: (file: File) => void;
  onSampleSelected: (sample: SampleItem) => void;
  samples: SampleItem[];
  isProcessing: boolean;
  className?: string;
}

export const AudioDropzone: React.FC<AudioDropzoneProps> = ({
  onFileSelected,
  onSampleSelected,
  samples,
  isProcessing,
  className,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [micLevel, setMicLevel] = useState(0);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const animFrameRef = useRef<number | null>(null);
  const timerRef = useRef<any>(null);

  // Keyboard shortcut: Press U to trigger file selection
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.key === 'u' || e.key === 'U') && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        e.preventDefault();
        fileInputRef.current?.click();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!isProcessing) setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (isProcessing) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndDispatch(file);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndDispatch(file);
    }
  };

  const validateAndDispatch = (file: File) => {
    const allowed = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'];
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowed.includes(ext)) {
      alert(`Unsupported format: ${ext}. Please upload a WAV, MP3, FLAC, OGG, or M4A file.`);
      return;
    }
    onFileSelected(file);
  };

  // Microphone recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const updateLevel = () => {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const avg = sum / bufferLength;
        setMicLevel(Math.min(100, (avg / 128) * 100));
        animFrameRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        stream.getTracks().forEach((t) => t.stop());
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const recordedFile = new File([audioBlob], `mic_recording_${Date.now()}.wav`, {
          type: 'audio/wav',
        });
        onFileSelected(recordedFile);
        setIsRecording(false);
        setRecordingSeconds(0);
        setMicLevel(0);
      };

      mediaRecorder.start();
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Could not access microphone. Please check your browser permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      clearInterval(timerRef.current);
      mediaRecorderRef.current.stop();
    }
  };

  return (
    <div className={clsx('flex flex-col gap-4', className)}>
      {/* Hidden Native File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".wav,.mp3,.flac,.ogg,.m4a"
        className="hidden"
        onChange={handleFileInputChange}
      />

      {/* Main Drag & Drop Card */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        className={clsx(
          'relative border-2 border-dashed rounded-[12px] p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors duration-150 select-none group',
          isDragOver
            ? 'border-primary bg-primary/5'
            : 'border-borderSubtle bg-surface hover:border-borderStrong hover:bg-elevated/40'
        )}
      >
        <div className="w-10 h-10 rounded-[8px] bg-primary/10 text-primary flex items-center justify-center mb-2.5 transition-colors">
          <Upload className="w-5 h-5" />
        </div>

        <h3 className="text-sm font-semibold text-txt-primary mb-1">
          Drop audio file here, or <span className="text-primary underline underline-offset-2">browse</span>
        </h3>
        <p className="text-xs text-txt-secondary mb-3">
          WAV, MP3, FLAC, M4A, or OGG · Up to 25 MB
        </p>

        <span className="text-[11px] font-mono text-txt-muted bg-elevated px-2 py-0.5 rounded-[6px] border border-borderSubtle">
          Press 'U' to upload
        </span>
      </div>

      {/* Action Row: Microphone Recorder */}
      <div className="flex items-center gap-2">
        {!isRecording ? (
          <Button
            variant="secondary"
            size="md"
            icon={<Mic className="w-4 h-4 text-txt-secondary" />}
            onClick={startRecording}
            disabled={isProcessing}
            className="w-full text-xs"
          >
            Record Live Voice
          </Button>
        ) : (
          <div className="w-full flex items-center justify-between p-2.5 bg-spoof-bg border border-spoof/30 rounded-[10px]">
            <div className="flex items-center gap-2.5">
              <span className="w-2.5 h-2.5 rounded-full bg-spoof animate-pulse" />
              <span className="text-xs font-medium text-spoof">
                Recording ({recordingSeconds}s)
              </span>
            </div>

            {/* Live Audio Level Meter */}
            <div className="w-24 h-2 bg-elevated rounded-full overflow-hidden border border-borderSubtle">
              <div
                className="h-full bg-primary transition-all duration-75"
                style={{ width: `${micLevel}%` }}
              />
            </div>

            <Button variant="danger" size="sm" icon={<MicOff className="w-3.5 h-3.5" />} onClick={stopRecording}>
              Stop
            </Button>
          </div>
        )}
      </div>

      {/* Reference Preset Samples */}
      {samples.length > 0 && (
        <div className="flex flex-col gap-2 pt-2 border-t border-borderSubtle">
          <span className="text-xs uppercase tracking-wide font-medium text-txt-secondary">
            Reference Samples
          </span>
          <div className="grid grid-cols-1 gap-2">
            {samples.map((sample) => (
              <button
                key={sample.id}
                onClick={() => onSampleSelected(sample)}
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
                      ? 'bg-genuine-bg border-genuine-border text-genuine'
                      : 'bg-spoof-bg border-spoof-border text-spoof'
                  )}
                >
                  {sample.ground_truth === 'bonafide' ? 'Genuine' : 'Deepfake'}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
