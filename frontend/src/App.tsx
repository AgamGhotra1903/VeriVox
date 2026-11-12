import React, { useState, useEffect } from 'react';
import { Header } from './components/layout/Header';
import { PrintableReport } from './components/layout/PrintableReport';
import { ToastProvider } from './components/ui/Toast';
import { AnalyseScreen } from './screens/AnalyseScreen';
import { AnalysisResult, HealthResponse, SampleItem } from './types/api';

export const App: React.FC = () => {
  const [activeResult, setActiveResult] = useState<AnalysisResult | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error('API offline');
        return res.json();
      })
      .then((data: HealthResponse) => {
        setHealth(data);
        setIsOnline(true);
      })
      .catch(() => {
        setIsOnline(false);
      });

    fetch('/api/samples')
      .then((res) => res.json())
      .then((data: SampleItem[]) => {
        setSamples(data);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return;
      if ((e.key === 'p' || e.key === 'P') && activeResult) {
        e.preventDefault();
        window.print();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeResult]);

  const handlePrint = () => {
    window.print();
  };

  return (
    <ToastProvider>
      <div className="min-h-screen bg-chassis text-txt-primary flex flex-col font-sans antialiased">
        {}
        <div className="print:hidden">
          <Header
            device={health?.device || 'MPS'}
            isOnline={isOnline}
            onPrintReport={handlePrint}
            canPrint={activeResult !== null}
          />
        </div>

        {}
        <main className="flex-1 print:hidden">
          <AnalyseScreen
            samples={samples}
            onAnalysisComplete={setActiveResult}
            activeResult={activeResult}
          />
        </main>

        {}
        {activeResult && <PrintableReport result={activeResult} />}

        {}
        <footer className="border-t border-borderSubtle bg-surface/50 py-2.5 px-4 text-[10px] font-mono text-txt-muted flex flex-col sm:flex-row items-center justify-between gap-2 print:hidden select-none">
          <div className="flex items-center gap-3">
            <span>ACOUSTIC FORENSICS LAB</span>
            <span className="w-1 h-1 rounded-full bg-borderStrong" />
            <span>CRNN + GMM + L2-STACKING</span>
            <span className="w-1 h-1 rounded-full bg-borderStrong" />
            <span>TRAINED ON ASVSPOOF 2019 LA</span>
          </div>

          <div className="flex items-center gap-4">
            <span>Shortcuts: <kbd className="text-phosphor">U</kbd> Upload · <kbd className="text-phosphor">Space</kbd> Play · <kbd className="text-phosphor">P</kbd> Print</span>
            <span className="w-1 h-1 rounded-full bg-borderStrong" />
            <span className="text-txt-secondary">VeriVox Forensic Lab</span>
          </div>
        </footer>
      </div>
    </ToastProvider>
  );
};

export default App;

