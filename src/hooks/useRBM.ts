import { useState, useCallback, useRef } from 'react';
import { BernoulliRBM } from '../ml/BernoulliRBM';
import type { TrainingProgress, DataSource, TrainingMethod } from '../types';
import { loadMNIST } from '../utils/mnistGenerator';
import { loadRealMNIST } from '../utils/mnistLoader';

export function useRBM() {
  const [rbm, setRBM] = useState<BernoulliRBM | null>(null);
  const [mnistData, setMnistData] = useState<Float32Array[] | null>(null);
  const [trainingData, setTrainingData] = useState<Float32Array[] | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [dataSource, setDataSource] = useState<DataSource>('generated');
  const [trainingMethod, setTrainingMethod] = useState<TrainingMethod>('contrastive-divergence');

  const abortController = useRef<AbortController | null>(null);

  const loadData = useCallback(async (forceReload = false) => {
    if (!mnistData || forceReload) {
      let data: Float32Array[];
      
      if (dataSource === 'mnist') {
        try {
          data = await loadRealMNIST();
        } catch (error) {
          console.error('Переключение на сгенерированные данные из-за ошибки:', error);
          data = await loadMNIST();
        }
      } else {
        data = await loadMNIST();
      }
      
      setMnistData(data);
      return data;
    }
    return mnistData;
  }, [mnistData, dataSource]);

  const trainNetwork = useCallback(async () => {
    console.log(`🎬 Начинаем обучение. Текущий метод: ${trainingMethod}`);
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

      const fullData = await loadData(true); // Принудительная перезагрузка при смене источника
      
      if (abortController.current?.signal.aborted) return;

      // Ограничиваем данные для имитации отжига
      let trainingData: Float32Array[];
      if (trainingMethod === 'simulated-annealing') {
        // Выбираем 10 разных образцов с разными цифрами
        const selectedIndices = new Set<number>();
        const digitCounts = new Array(10).fill(0); // Счетчик для каждой цифры (0-9)
        
        // Сначала пытаемся найти образцы с разными цифрами
        for (let i = 0; i < fullData.length && selectedIndices.size < 10; i++) {
          // Простая эвристика для определения цифры по паттерну
          let digit = Math.floor(Math.random() * 10); // Временное решение
          
          // Если у нас еще нет этой цифры или у нас меньше 10 образцов
          if (digitCounts[digit] === 0 || selectedIndices.size < 5) {
            selectedIndices.add(i);
            digitCounts[digit]++;
          }
        }
        
        // Если не нашли достаточно разных, добавляем случайные
        while (selectedIndices.size < 10) {
          const randomIndex = Math.floor(Math.random() * fullData.length);
          if (!selectedIndices.has(randomIndex)) {
            selectedIndices.add(randomIndex);
          }
        }
        
        trainingData = Array.from(selectedIndices).map(index => fullData[index]);
        console.log(`❄️ Имитация отжига: выбрано ${trainingData.length} разных образцов из ${fullData.length}`);
      } else {
        trainingData = fullData;
      }

      setTrainingData(trainingData);

      setTrainingProgress({
        epoch: 0,
        totalEpochs: 15,
        progress: 10,
        status: 'Инициализация RBM...'
      });

      console.log(`🎯 Создаем RBM с методом обучения: ${trainingMethod}`);
      const newRBM = new BernoulliRBM({
        nVisible: 784,
        nHidden: 64,
        learningRate: 0.06,
        batchSize: 32,
        trainingMethod: trainingMethod
      });

      if (abortController.current?.signal.aborted) return;

      // Увеличиваем количество эпох для имитации отжига
      const epochs = trainingMethod === 'simulated-annealing' ? 25 : 15;
      await newRBM.fit(trainingData, epochs, (epoch, totalEpochs) => {
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
  }, [isTraining, loadData, trainingMethod]);

  const loadSavedWeights = useCallback(() => {
    const loadedRBM = BernoulliRBM.loadFromLocalStorage();
    if (loadedRBM) {
      setRBM(loadedRBM);
      loadData(true);
    }
  }, [loadData]);

  const handleDataSourceChange = useCallback((newDataSource: DataSource) => {
    setDataSource(newDataSource);
    setMnistData(null); // Очищаем кеш данных при смене источника
  }, []);

  const handleTrainingMethodChange = useCallback((newMethod: TrainingMethod) => {
    console.log(`🔄 Смена метода обучения: ${trainingMethod} → ${newMethod}`);
    setTrainingMethod(newMethod);
  }, [trainingMethod]);

  const stopTraining = useCallback(() => {
    if (abortController.current) {
      abortController.current.abort();
      setIsTraining(false);
      setTrainingProgress(null);
    }
  }, []);

  // Возвращаем данные для визуализации в зависимости от метода обучения
  const getVisualizationData = useCallback(() => {
    if (trainingMethod === 'simulated-annealing' && trainingData) {
      return trainingData; // Для SA показываем только данные обучения
    }
    return mnistData; // Для CD показываем все данные
  }, [trainingMethod, trainingData, mnistData]);

  return {
    rbm,
    mnistData: getVisualizationData(),
    isTraining,
    trainingProgress,
    dataSource,
    trainingMethod,
    trainNetwork,
    loadSavedWeights,
    stopTraining,
    loadData,
    handleDataSourceChange,
    handleTrainingMethodChange
  };
}