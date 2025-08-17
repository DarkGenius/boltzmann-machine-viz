import { useState, useCallback, useRef } from 'react';
import { BernoulliRBM } from '../ml/BernoulliRBM';
import type { TrainingProgress, DataSource, TrainingMethod } from '../types';
import { loadMNIST } from '../utils/mnistGenerator';
import { loadRealMNIST } from '../utils/mnistLoader';
import { DEFAULT_SAMPLE_DIGIT, DEFAULT_EPOCHS } from '../constants';

export function useRBM() {
  const [rbm, setRBM] = useState<BernoulliRBM | null>(null);
  const [mnistData, setMnistData] = useState<Float32Array[] | null>(null);
  const [trainingData, setTrainingData] = useState<Float32Array[] | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [dataSource, setDataSource] = useState<DataSource>('generated');
  const [trainingMethod, setTrainingMethod] = useState<TrainingMethod>('contrastive-divergence');
  const [selectedDigit, setSelectedDigit] = useState(DEFAULT_SAMPLE_DIGIT);
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);

  const abortController = useRef<AbortController | null>(null);

  const loadData = useCallback(async (forceReload = false) => {
    if (!mnistData || forceReload) {
      let data: Float32Array[];

      if (dataSource === 'mnist') {
        try {
          if (trainingMethod === 'equilibrium') {
            // Для equilibrium sampling отбираем конкретную цифру из MNIST
            console.log(`📊 Загружаем реальные данные MNIST для цифры: ${selectedDigit}`);
            const { loadRealMNIST, getMNISTLabels } = await import('../utils/mnistLoader');
            const [mnistData, labels] = await Promise.all([
              loadRealMNIST(),
              getMNISTLabels()
            ]);

            // Фильтруем данные только для выбранной цифры
            const filteredData = mnistData.filter((_, index) => labels[index] === selectedDigit);
            data = filteredData.slice(0, 20); // Берем первые 20 образцов
            console.log(`✅ Отфильтровано ${data.length} образцов цифры ${selectedDigit} из реальных данных MNIST`);
          } else {
            // Для CD используем все данные MNIST
            console.log(`📊 Загружаем все реальные данные MNIST для CD`);
            data = await loadRealMNIST();
          }
        } catch (error) {
          console.error('Переключение на сгенерированные данные из-за ошибки:', error);
          if (trainingMethod === 'equilibrium') {
            data = await loadMNIST(20, [selectedDigit]);
          } else {
            data = await loadMNIST();
          }
        }
      } else {
        if (trainingMethod === 'equilibrium') {
          console.log(`📊 Загружаем сгенерированные данные для цифры: ${selectedDigit}`);
          data = await loadMNIST(20, [selectedDigit]);
        } else {
          console.log(`📊 Загружаем все сгенерированные данные для CD`);
          data = await loadMNIST();
        }
      }

      setMnistData(data);
      return data;
    }
    return mnistData;
  }, [mnistData, dataSource, trainingMethod, selectedDigit]);

  const trainNetwork = useCallback(async () => {
    console.log(`🎬 Начинаем обучение. Текущий метод: ${trainingMethod}`);
    if (isTraining) return;

    setIsTraining(true);
    abortController.current = new AbortController();

    try {
      setTrainingProgress({
        epoch: 0,
        totalEpochs: epochs,
        progress: 0,
        status: 'Загрузка данных...'
      });

      const fullData = await loadData(true); // Принудительная перезагрузка при смене источника

      if (abortController.current?.signal.aborted) return;

      // Используем данные в зависимости от метода
      let trainingData: Float32Array[] = fullData;
      if (trainingMethod === 'equilibrium') {
        console.log(`❄️ Equilibrium sampling: используем ${trainingData.length} образцов цифры ${selectedDigit}`);
      } else {
        console.log(`⚡ Contrastive Divergence: используем ${trainingData.length} образцов`);
      }

      setTrainingData(trainingData);

      setTrainingProgress({
        epoch: 0,
        totalEpochs: epochs,
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

      // Используем настройку количества эпох
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
        epoch: epochs,
        totalEpochs: epochs,
        progress: 100,
        status: 'Обучение завершено!'
      });

    } catch (error) {
      console.error('Ошибка обучения:', error);
      setTrainingProgress({
        epoch: 0,
        totalEpochs: epochs,
        progress: 0,
        status: 'Ошибка обучения'
      });
    } finally {
      setIsTraining(false);
      setTimeout(() => setTrainingProgress(null), 2000);
    }
  }, [isTraining, loadData, trainingMethod, selectedDigit, epochs]);

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

  const handleSelectedDigitChange = useCallback((digit: number) => {
    console.log(`🔢 Смена выбранной цифры: ${selectedDigit} → ${digit}`);
    setSelectedDigit(digit);
    setMnistData(null); // Очищаем кеш данных при смене цифры
  }, [selectedDigit]);

  const handleEpochsChange = useCallback((newEpochs: number) => {
    console.log(`🔄 Смена количества эпох: ${epochs} → ${newEpochs}`);
    setEpochs(newEpochs);
  }, [epochs]);

  const stopTraining = useCallback(() => {
    if (abortController.current) {
      abortController.current.abort();
      setIsTraining(false);
      setTrainingProgress(null);
    }
  }, []);

  // Возвращаем данные для визуализации
  const getVisualizationData = useCallback(() => {
    return mnistData; // Показываем все данные
  }, [mnistData]);

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
    handleTrainingMethodChange,
    handleSelectedDigitChange,
    handleEpochsChange
  };
}