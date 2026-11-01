import { useState, useEffect, useCallback } from 'react';
import { HealthStatus } from '../types';
import { apiService } from '../services/api';

export function useHealth(pollIntervalMs: number = 15000) {
  const [health, setHealth] = useState<HealthStatus>({
    status: 'offline',
    model_loaded: false,
    device: 'unknown',
    version: '1.0.0',
    architecture: 'ResNet50 + BERT + BAM',
  });
  const [loading, setLoading] = useState<boolean>(true);

  const check = useCallback(async () => {
    const res = await apiService.getHealth();
    setHealth(res);
    setLoading(false);
  }, []);

  useEffect(() => {
    check();
    const interval = setInterval(check, pollIntervalMs);
    return () => clearInterval(interval);
  }, [check, pollIntervalMs]);

  return { health, loading, refreshHealth: check };
}
