import { Menu, RefreshCw } from 'lucide-react';
import type { TabType, HealthStatus } from '../../types';

interface HeaderProps {
  activeTab: TabType;
  health: HealthStatus;
  refreshHealth: () => void;
  setMobileOpen: (open: boolean) => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  health,
  refreshHealth,
  setMobileOpen,
}) => {
  const titles: Record<TabType, { title: string; subtitle: string }> = {
    overview: {
      title: 'Overview & Capabilities',
      subtitle: 'Multimodal content moderation powered by deep cross-attention fusion',
    },
    moderation: {
      title: 'Content Moderation Studio',
      subtitle: 'Submit an image and optional caption for live multimodal analysis',
    },
    history: {
      title: 'Analysis History',
      subtitle: 'Review recently analyzed meme samples and inference metadata',
    },
    about: {
      title: 'About MemeGuard AI',
      subtitle: 'Multimodal social media content analysis, features, and responsible AI guidance',
    },
  };

  const current = titles[activeTab] || titles.moderation;

  return (
    <header className="sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200 px-4 sm:px-8 py-3.5 flex items-center justify-between">
      <div className="flex items-center space-x-3 sm:space-x-4">
        <button
          onClick={() => setMobileOpen(true)}
          className="lg:hidden p-2 text-slate-500 hover:text-slate-900 rounded-lg hover:bg-slate-100"
          aria-label="Open Navigation Menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div>
          <h1 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight flex items-center space-x-2">
            <span>{current.title}</span>
          </h1>
          <p className="hidden sm:block text-xs text-slate-500 mt-0.5 font-normal">
            {current.subtitle}
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        {/* Model Availability Badge */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200/80 text-xs text-slate-600">
          <span className="relative flex h-2 w-2">
            {health.model_loaded && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            )}
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                health.model_loaded ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            ></span>
          </span>
          <span className="hidden md:inline font-medium">
            {health.model_loaded ? 'Model Live' : 'Backend Offline'}
          </span>
          <span className="text-slate-400 font-mono text-[10px]">
            ({health.device.toUpperCase()})
          </span>
          <button
            onClick={refreshHealth}
            title="Refresh Connection"
            className="text-slate-400 hover:text-indigo-600 transition-colors ml-1 cursor-pointer"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>
      </div>
    </header>
  );
};
