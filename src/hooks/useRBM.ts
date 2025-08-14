import { useState, useCallback, useRef } from 'react';
import { BernoulliRBM } from '../ml/BernoulliRBM';
import type { TrainingProgress } from '../types';
import { loadMNIST } from '../utils/mnistGenerator';

export function useRBM() {
  const [rbm, setRBM] = useState<BernoulliRBM | null>(null);
  const [mnistData, setMnistData] = useState<Float32Array[] | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);

  const abortController = useRef<AbortController | null>(null);

  const loadData = useCallback(async () => {
    if (!mnistData) {
      const data = await loadMNIST();
      setMnistData(data);
      return data;
    }
    return mnistData;
  }, [mnistData]);

  const trainNetwork = useCallback(async () => {
    if (isTraining) return;

    setIsTraining(true);
    abortController.current = new AbortController();
    
    try {
      setTrainingProgress({
        epoch: 0,
        totalEpochs: 15,
        progress: 0,
        status: 'Загрузка данных...'
      });

      const data = await loadData();
      
      if (abortController.current?.signal.aborted) return;

      setTrainingProgress({
        epoch: 0,
        totalEpochs: 15,
        progress: 10,
        status: 'Инициализация RBM...'
      });

      const newRBM = new BernoulliRBM({
        nVisible: 784,
        nHidden: 64,
        learningRate: 0.06,
        batchSize: 32
      });

      if (abortController.current?.signal.aborted) return;

      await newRBM.fit(data, 15, (epoch, totalEpochs) => {
        if (abortController.current?.signal.aborted) return;
        
        const progress = 10 + (epoch / totalEpochs) * 90;
        setTrainingProgress({
          epoch,
          totalEpochs,
          progress,
          status: `Обучение сети... Эпоха ${epoch}/${totalEpochs}`
        });
      });

      if (abortController.current?.signal.aborted) return;

      setRBM(newRBM);
      
      const saveWeights = localStorage.getItem('rbm_save_weights') !== 'false';
      if (saveWeights) {
        newRBM.saveToLocalStorage();
      }

      setTrainingProgress({
        epoch: 15,
        totalEpochs: 15,
        progress: 100,
        status: 'Обучение завершено!'
      });

    } catch (error) {
      console.error('Ошибка обучения:', error);
      setTrainingProgress({
        epoch: 0,
        totalEpochs: 15,
        progress: 0,
        status: 'Ошибка обучения'
      });
    } finally {
      setIsTraining(false);
      setTimeout(() => setTrainingProgress(null), 2000);
    }
  }, [isTraining, loadData]);

  const loadSavedWeights = useCallback(() => {
    const loadedRBM = BernoulliRBM.loadFromLocalStorage();
    if (loadedRBM) {
      setRBM(loadedRBM);
      loadData();
    }
  }, [loadData]);

  const stopTraining = useCallback(() => {
    if (abortController.current) {
      abortController.current.abort();
      setIsTraining(false);
      setTrainingProgress(null);
    }
  }, []);

  return {
    rbm,
    mnistData,
    isTraining,
    trainingProgress,
    trainNetwork,
    loadSavedWeights,
    stopTraining,
    loadData
  };
}