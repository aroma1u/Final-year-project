import { useState, useEffect } from 'react';
import { PredictionResult } from '../types';

const STORAGE_KEY = 'memeguard_local_history_v1';

export function useHistory() {
  const [history, setHistory] = useState<PredictionResult[]>([]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setHistory(JSON.parse(stored));
      }
    } catch (e) {
      console.warn('[useHistory] Could not read history from localStorage:', e);
    }
  }, []);

  const addHistoryItem = (item: PredictionResult) => {
    setHistory((prev) => {
      // Limit to latest 30 items
      const updated = [item, ...prev].slice(0, 30);
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      } catch (e) {
        console.warn('[useHistory] Could not write history to localStorage:', e);
      }
      return updated;
    });
  };

  const clearHistory = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
      setHistory([]);
    } catch (e) {
      console.warn('[useHistory] Could not clear localStorage:', e);
    }
  };

  const removeItem = (id: string) => {
    setHistory((prev) => {
      const updated = prev.filter((item) => item.id !== id);
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      } catch (e) {
        console.warn('[useHistory] Could not update localStorage:', e);
      }
      return updated;
    });
  };

  return { history, addHistoryItem, clearHistory, removeItem };
}
