import React from 'react';
import { 
  ShieldCheck, 
  LayoutDashboard, 
  ScanEye, 
  History, 
  Info, 
  CheckCircle2, 
  AlertCircle, 
  XCircle, 
  X,
  Server
} from 'lucide-react';
import type { TabType, HealthStatus } from '../../types';

interface SidebarProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  health: HealthStatus;
  mobileOpen: boolean;
  setMobileOpen: (open: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  health,
  mobileOpen,
  setMobileOpen,
}) => {
  const navItems: { id: TabType; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'overview', label: 'Overview', icon: <LayoutDashboard className="w-5 h-5" /> },
    { id: 'moderation', label: 'Content Moderation', icon: <ScanEye className="w-5 h-5" />, badge: 'Studio' },
    { id: 'history', label: 'Analysis History', icon: <History className="w-5 h-5" /> },
    { id: 'about', label: 'About This Website', icon: <Info className="w-5 h-5" /> },
  ];

  const handleNavClick = (tab: TabType) => {
    setActiveTab(tab);
    setMobileOpen(false);
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-40 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 bg-white border-r border-slate-200 flex flex-col transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand Header */}
        <div className="p-5 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-700 to-indigo-900 flex items-center justify-center text-white shadow-md shadow-indigo-100">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="font-bold text-slate-900 tracking-tight flex items-center space-x-1">
                <span>MemeGuard</span>
                <span className="text-indigo-600 font-extrabold text-xs px-1.5 py-0.5 bg-indigo-50 rounded-md">AI</span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium">MSCMGTB Capstone</p>
            </div>
          </div>
          <button 
            onClick={() => setMobileOpen(false)}
            className="lg:hidden p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation List */}
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          <div className="px-3 pt-2 pb-1.5 text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
            Platform Menu
          </div>
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleNavClick(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-200'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className={isActive ? 'text-white' : 'text-slate-400 group-hover:text-slate-600'}>
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                      isActive
                        ? 'bg-white/20 text-white'
                        : 'bg-indigo-50 text-indigo-700'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Model Status Card */}
        <div className="p-4 border-t border-slate-100 bg-slate-50/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-700 flex items-center space-x-1.5">
              <Server className="w-3.5 h-3.5 text-slate-400" />
              <span>Inference Engine</span>
            </span>
            {health.status === 'healthy' ? (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100 text-emerald-800">
                <CheckCircle2 className="w-3 h-3 mr-1" /> Ready
              </span>
            ) : health.status === 'degraded' ? (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-100 text-amber-800">
                <AlertCircle className="w-3 h-3 mr-1" /> Degraded
              </span>
            ) : (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-100 text-rose-800">
                <XCircle className="w-3 h-3 mr-1" /> Offline
              </span>
            )}
          </div>
          <div className="space-y-1 text-[11px] text-slate-500">
            <div className="flex justify-between">
              <span className="text-slate-400">Backbone:</span>
              <span className="font-medium text-slate-700">CNN + BERT + BAM</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Device:</span>
              <span className="font-mono text-slate-700 uppercase">{health.device}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Threshold (τ*):</span>
              <span className="font-mono font-medium text-indigo-600">0.41</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
