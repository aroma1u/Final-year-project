import { useState } from 'react';
import type { TabType } from './types';
import { useHealth } from './hooks/useHealth';
import { useHistory } from './hooks/useHistory';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { OverviewPage } from './pages/OverviewPage';
import { ModerationPage } from './pages/ModerationPage';
import { HistoryPage } from './pages/HistoryPage';
import { AboutPage } from './pages/AboutPage';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('moderation');
  const [mobileOpen, setMobileOpen] = useState<boolean>(false);
  const { health, refreshHealth } = useHealth(15000);
  const { history, addHistoryItem, clearHistory, removeItem } = useHistory();

  return (
    <div className="min-h-screen bg-slate-50 font-sans flex text-slate-800 antialiased selection:bg-indigo-100 selection:text-indigo-900">
      {/* Persistent Left Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 lg:pl-64">
        {/* Sticky Top Header */}
        <Header
          activeTab={activeTab}
          health={health}
          refreshHealth={refreshHealth}
          setMobileOpen={setMobileOpen}
        />

        {/* Dynamic Page Router */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto">
          {activeTab === 'overview' && (
            <OverviewPage setActiveTab={setActiveTab} health={health} />
          )}
          {activeTab === 'moderation' && (
            <ModerationPage health={health} onPredictionComplete={addHistoryItem} />
          )}
          {activeTab === 'history' && (
            <HistoryPage
              history={history}
              clearHistory={clearHistory}
              removeItem={removeItem}
              setActiveTab={setActiveTab}
            />
          )}
          {activeTab === 'about' && <AboutPage />}
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200/80 bg-white/60 py-4 px-6 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            <span className="font-semibold text-slate-700">MemeGuard AI</span> &bull; Multimodal Social Media Content Moderation
          </div>
          <div className="flex items-center space-x-4 text-[11px] text-slate-400">
            <span>B.Tech Final Year CSE Capstone</span>
            <span>&bull;</span>
            <span>PyTorch ResNet-50 + BERT + BAM</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
