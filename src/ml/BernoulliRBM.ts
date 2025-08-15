import type { RBMParams, ReconstructionResult, TrainingMethod } from '../types';

export class BernoulliRBM {
  private nVisible: number;
  private nHidden: number;
  private learningRate: number;
  private batchSize: number;
  private trainingMethod: TrainingMethod;
  private weights: Float32Array[];
  private hiddenBias: Float32Array;
  private visibleBias: Float32Array;

  constructor({ nVisible, nHidden, learningRate = 0.06, batchSize = 32, trainingMethod = 'contrastive-divergence' }: RBMParams) {
    this.nVisible = nVisible;
    this.nHidden = nHidden;
    this.learningRate = learningRate;
    this.batchSize = batchSize;
    this.trainingMethod = trainingMethod;

    this.weights = this.randomMatrix(nHidden, nVisible, 0.01);
    this.hiddenBias = new Float32Array(nHidden);
    this.visibleBias = new Float32Array(nVisible);
  }

  private randomMatrix(rows: number, cols: number, scale: number): Float32Array[] {
    const matrix: Float32Array[] = [];
    for (let i = 0; i < rows; i++) {
      matrix[i] = new Float32Array(cols);
      for (let j = 0; j < cols; j++) {
        matrix[i][j] = (Math.random() - 0.5) * 2 * scale;
      }
    }
    return matrix;
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  // Вычисление энергии системы (функция Ляпунова)
  private computeEnergy(visible: Float32Array, hidden: Float32Array): number {
    let energy = 0;
    
    // Член смещения видимого слоя
    for (let i = 0; i < this.nVisible; i++) {
      energy -= this.visibleBias[i] * visible[i];
    }
    
    // Член смещения скрытого слоя
    for (let j = 0; j < this.nHidden; j++) {
      energy -= this.hiddenBias[j] * hidden[j];
    }
    
    // Член взаимодействия между слоями
    for (let i = 0; i < this.nVisible; i++) {
      for (let j = 0; j < this.nHidden; j++) {
        energy -= this.weights[j][i] * visible[i] * hidden[j];
      }
    }
    
    return energy;
  }

  /**
   * Выборка Гиббса - основной метод для генерации новых состояний в RBM
   * Чередует выборку видимого и скрытого слоев для получения образцов из распределения
   */
  private gibbsSample(visible: Float32Array, steps: number = 1): { visible: Float32Array, hidden: Float32Array } {
    let currentVisible = visible.slice();
    let currentHidden = new Float32Array(this.nHidden);
    
    for (let step = 0; step < steps; step++) {
      // Шаг 1: Выборка скрытого слоя на основе видимого
      currentHidden = this.sampleHiddenBinary(currentVisible);
      
      // Шаг 2: Выборка видимого слоя на основе скрытого
      currentVisible = this.sampleVisibleBinary(currentHidden);
    }
    
    return { visible: currentVisible, hidden: currentHidden };
  }
  
  /**
   * Бинарная выборка скрытого слоя (используется для выборки Гиббса)
   * Возвращает бинарные активации вместо вероятностей
   */
  private sampleHiddenBinary(visible: Float32Array): Float32Array {
    const hidden = new Float32Array(this.nHidden);
    for (let i = 0; i < this.nHidden; i++) {
      let activation = this.hiddenBias[i];
      for (let j = 0; j < this.nVisible; j++) {
        activation += visible[j] * this.weights[i][j];
      }
      const prob = this.sigmoid(activation);
      // Бинарная выборка: 1 с вероятностью prob, 0 иначе
      hidden[i] = Math.random() < prob ? 1 : 0;
    }
    return hidden;
  }
  
  /**
   * Бинарная выборка видимого слоя (используется для выборки Гиббса)
   * Возвращает бинарные активации вместо вероятностей
   */
  private sampleVisibleBinary(hidden: Float32Array): Float32Array {
    const visible = new Float32Array(this.nVisible);
    for (let i = 0; i < this.nVisible; i++) {
      let activation = this.visibleBias[i];
      for (let j = 0; j < this.nHidden; j++) {
        activation += hidden[j] * this.weights[j][i];
      }
      const prob = this.sigmoid(activation);
      // Бинарная выборка: 1 с вероятностью prob, 0 иначе
      visible[i] = Math.random() < prob ? 1 : 0;
    }
    return visible;
  }

  /**
   * Упрощенный метод имитации отжига с базовым алгоритмом Метрополиса-Гастингса
   * Фокус на стабильности и корректности принятия состояний
   */
  private simulatedAnnealing(data: Float32Array[]): void {
    console.log('🔥 Начинаем упрощенную имитацию отжига...');
    
    // Простые и стабильные параметры
    const initialTemperature = 2.0;
    const finalTemperature = 0.1;
    const coolingRate = 0.99;
    const stepsPerTemperature = 50;
    
    let temperature = initialTemperature;
    let totalIterations = 0;
    let acceptedMoves = 0;
    let energyDecreases = 0;
    
    // Начинаем с случайного образца
    let currentSample = data[Math.floor(Math.random() * data.length)].slice();
    let currentHidden = this.sampleHiddenBinary(currentSample);
    let currentEnergy = this.computeEnergy(currentSample, currentHidden);
    
    console.log(`🎯 Начальная энергия: ${currentEnergy.toFixed(4)}, температура: ${temperature.toFixed(4)}`);
    
    while (temperature > finalTemperature) {
      let tempAccepted = 0;
      
      for (let step = 0; step < stepsPerTemperature; step++) {
        totalIterations++;
        
        // Генерируем новое состояние: делаем небольшие изменения к текущему состоянию
        const proposedSample = currentSample.slice();
        const proposedHidden = currentHidden.slice();
        
        // Изменяем случайно выбранные нейроны (меньше изменений = выше принятие)
        const neuronsToFlip = Math.min(3, this.nHidden); // Максимум 3 нейрона
        for (let flip = 0; flip < neuronsToFlip; flip++) {
          const neuronIdx = Math.floor(Math.random() * this.nHidden);
          proposedHidden[neuronIdx] = proposedHidden[neuronIdx] > 0.5 ? 0 : 1;
        }
        
        // Перевычисляем видимый слой для нового скрытого состояния
        for (let i = 0; i < this.nVisible; i++) {
          let activation = this.visibleBias[i];
          for (let j = 0; j < this.nHidden; j++) {
            activation += proposedHidden[j] * this.weights[j][i];
          }
          const prob = this.sigmoid(activation);
          proposedSample[i] = Math.random() < prob ? 1 : 0;
        }
        
        const proposedEnergy = this.computeEnergy(proposedSample, proposedHidden);
        const energyDiff = proposedEnergy - currentEnergy;
        
        // Проверка корректности энергий
        if (!isFinite(currentEnergy) || !isFinite(proposedEnergy)) {
          console.error(`❌ Некорректная энергия на итерации ${totalIterations}`);
          continue;
        }
        
        // Отладка для первых итераций
        if (totalIterations <= 10) {
          console.log(`🔍 Итерация ${totalIterations}: E=${currentEnergy.toFixed(4)} → ${proposedEnergy.toFixed(4)}, ΔE=${energyDiff.toFixed(4)}`);
        }
        
        // Критерий Метрополиса: принимаем если энергия уменьшилась ИЛИ с вероятностью exp(-ΔE/T)
        let shouldAccept = false;
        if (energyDiff <= 0) {
          shouldAccept = true; // Всегда принимаем улучшения
        } else {
          const acceptProbability = Math.exp(-energyDiff / temperature);
          shouldAccept = Math.random() < acceptProbability;
          
          if (totalIterations <= 10) {
            console.log(`🎲 P(accept)=${(acceptProbability * 100).toFixed(1)}%, принят: ${shouldAccept ? 'Да' : 'Нет'}`);
          }
        }
        
        if (shouldAccept) {
          acceptedMoves++;
          tempAccepted++;
          
          if (energyDiff < 0) {
            energyDecreases++;
          }
          
          // Обновляем текущее состояние
          currentSample = proposedSample;
          currentHidden = proposedHidden;
          currentEnergy = proposedEnergy;
          
          // Обновление весов с дополнительной диагностикой
          if (acceptedMoves % 10 === 0) {
            const dataIdx = Math.floor(Math.random() * data.length);
            const dataSample = data[dataIdx];
            const dataHidden = this.sampleHidden(dataSample);
            
            const lr = this.learningRate * 0.01; // ОЧЕНЬ маленькая скорость обучения
            
            // Вычисляем статистики перед обновлением
            let totalWeightMagnitude = 0;
            let maxWeight = -Infinity;
            let minWeight = Infinity;
            
            for (let i = 0; i < this.nVisible; i++) {
              for (let j = 0; j < this.nHidden; j++) {
                const w = this.weights[j][i];
                totalWeightMagnitude += Math.abs(w);
                maxWeight = Math.max(maxWeight, w);
                minWeight = Math.min(minWeight, w);
              }
            }
            
            console.log(`🔧 Обновление весов #${Math.floor(acceptedMoves / 10)}:`);
            console.log(`   Веса до: среднее=${(totalWeightMagnitude / (this.nVisible * this.nHidden)).toFixed(4)}, мин=${minWeight.toFixed(4)}, макс=${maxWeight.toFixed(4)}`);
            
            // Обновляем веса с ограничением
            let totalUpdate = 0;
            let maxUpdate = 0;
            
            for (let i = 0; i < this.nVisible; i++) {
              for (let j = 0; j < this.nHidden; j++) {
                const dataCorr = dataSample[i] * dataHidden[j];
                const modelCorr = currentSample[i] * currentHidden[j];
                const update = lr * (dataCorr - modelCorr);
                
                // КРИТИЧНО: Ограничиваем размер обновления
                const clampedUpdate = Math.max(-0.01, Math.min(0.01, update));
                this.weights[j][i] += clampedUpdate;
                
                // КРИТИЧНО: Ограничиваем размер весов
                this.weights[j][i] = Math.max(-2.0, Math.min(2.0, this.weights[j][i]));
                
                totalUpdate += Math.abs(clampedUpdate);
                maxUpdate = Math.max(maxUpdate, Math.abs(clampedUpdate));
              }
              
              const biasUpdate = lr * (dataSample[i] - currentSample[i]);
              const clampedBiasUpdate = Math.max(-0.01, Math.min(0.01, biasUpdate));
              this.visibleBias[i] += clampedBiasUpdate;
              this.visibleBias[i] = Math.max(-1.0, Math.min(1.0, this.visibleBias[i]));
            }
            
            for (let j = 0; j < this.nHidden; j++) {
              const biasUpdate = lr * (dataHidden[j] - currentHidden[j]);
              const clampedBiasUpdate = Math.max(-0.01, Math.min(0.01, biasUpdate));
              this.hiddenBias[j] += clampedBiasUpdate;
              this.hiddenBias[j] = Math.max(-1.0, Math.min(1.0, this.hiddenBias[j]));
            }
            
            // Диагностика после обновления
            totalWeightMagnitude = 0;
            maxWeight = -Infinity;
            minWeight = Infinity;
            
            for (let i = 0; i < this.nVisible; i++) {
              for (let j = 0; j < this.nHidden; j++) {
                const w = this.weights[j][i];
                totalWeightMagnitude += Math.abs(w);
                maxWeight = Math.max(maxWeight, w);
                minWeight = Math.min(minWeight, w);
              }
            }
            
            console.log(`   Веса после: среднее=${(totalWeightMagnitude / (this.nVisible * this.nHidden)).toFixed(4)}, мин=${minWeight.toFixed(4)}, макс=${maxWeight.toFixed(4)}`);
            console.log(`   Размер обновлений: среднее=${(totalUpdate / (this.nVisible * this.nHidden)).toFixed(6)}, макс=${maxUpdate.toFixed(6)}`);
            
            // Тестируем энергию после обновления
            const testEnergy = this.computeEnergy(currentSample, currentHidden);
            console.log(`   Энергия после обновления весов: ${currentEnergy.toFixed(4)} → ${testEnergy.toFixed(4)} (изменение: ${(testEnergy - currentEnergy).toFixed(4)})`);
            currentEnergy = testEnergy; // Обновляем текущую энергию
          }
        }
      }
      
      // Охлаждение
      temperature *= coolingRate;
      
      // Статистика каждую тысячу итераций
      if (totalIterations % 1000 === 0) {
        const overallAcceptanceRate = (acceptedMoves / totalIterations * 100).toFixed(1);
        const tempAcceptanceRate = (tempAccepted / stepsPerTemperature * 100).toFixed(1);
        console.log(`🌡️ Итерация ${totalIterations}, T=${temperature.toFixed(3)}, принято: ${overallAcceptanceRate}% (последние ${tempAcceptanceRate}%)`);
        console.log(`⚡ Текущая энергия: ${currentEnergy.toFixed(4)}`);
      }
    }
    
    // Финальная статистика
    const finalAcceptanceRate = (acceptedMoves / totalIterations * 100).toFixed(1);
    const improvementRate = acceptedMoves > 0 ? (energyDecreases / acceptedMoves * 100).toFixed(1) : '0';
    
    console.log(`❄️ Имитация отжига завершена за ${totalIterations} итераций`);
    console.log(`📊 Общий процент принятия: ${finalAcceptanceRate}%`);
    console.log(`📈 Из принятых состояний, улучшений: ${improvementRate}%`);
    console.log(`⚡ Финальная энергия: ${currentEnergy.toFixed(4)}`);
    console.log(`🎯 Обновлений весов: ${Math.floor(acceptedMoves / 10)}`);
  }

  private sampleHidden(visible: Float32Array): Float32Array {
    const hidden = new Float32Array(this.nHidden);
    for (let i = 0; i < this.nHidden; i++) {
      let activation = this.hiddenBias[i];
      for (let j = 0; j < this.nVisible; j++) {
        activation += visible[j] * this.weights[i][j];
      }
      hidden[i] = this.sigmoid(activation);
    }
    return hidden;
  }

  private sampleVisible(hidden: Float32Array): Float32Array {
    const visible = new Float32Array(this.nVisible);
    for (let i = 0; i < this.nVisible; i++) {
      let activation = this.visibleBias[i];
      for (let j = 0; j < this.nHidden; j++) {
        activation += hidden[j] * this.weights[j][i];
      }
      visible[i] = this.sigmoid(activation);
    }
    return visible;
  }

  private contrastiveDivergence(batch: Float32Array[]): void {
    const batchSize = batch.length;
    
    const weightGrad = this.randomMatrix(this.nHidden, this.nVisible, 0);
    const hiddenGrad = new Float32Array(this.nHidden);
    const visibleGrad = new Float32Array(this.nVisible);

    for (const sample of batch) {
      const hiddenProb = this.sampleHidden(sample);
      const visibleRecon = this.sampleVisible(hiddenProb);
      const hiddenRecon = this.sampleHidden(visibleRecon);

      for (let i = 0; i < this.nHidden; i++) {
        for (let j = 0; j < this.nVisible; j++) {
          weightGrad[i][j] += (hiddenProb[i] * sample[j] - hiddenRecon[i] * visibleRecon[j]) / batchSize;
        }
        hiddenGrad[i] += (hiddenProb[i] - hiddenRecon[i]) / batchSize;
      }

      for (let i = 0; i < this.nVisible; i++) {
        visibleGrad[i] += (sample[i] - visibleRecon[i]) / batchSize;
      }
    }

    for (let i = 0; i < this.nHidden; i++) {
      for (let j = 0; j < this.nVisible; j++) {
        this.weights[i][j] += this.learningRate * weightGrad[i][j];
      }
      this.hiddenBias[i] += this.learningRate * hiddenGrad[i];
    }

    for (let i = 0; i < this.nVisible; i++) {
      this.visibleBias[i] += this.learningRate * visibleGrad[i];
    }
  }

  async fit(
    data: Float32Array[], 
    nEpochs = 15, 
    progressCallback?: (epoch: number, totalEpochs: number) => void
  ): Promise<void> {
    console.log(`🚀 Начинаем обучение методом: ${this.trainingMethod}`);
    
    if (this.trainingMethod === 'simulated-annealing') {
      console.log('❄️ Используем имитацию отжига');
      // Для имитации отжига используем меньше данных, но больше эпох
      const reducedData = data.slice(0, Math.min(10, data.length));
      const reducedEpochs = Math.min(25, nEpochs); // Увеличиваем количество эпох
      console.log(`📊 Данные: ${reducedData.length} образцов, эпох: ${reducedEpochs}`);
      
      for (let epoch = 0; epoch < reducedEpochs; epoch++) {
        if (progressCallback) {
          progressCallback(epoch + 1, reducedEpochs);
        }
        
        // Логирование прогресса обучения
        if (epoch > 0 && epoch % 5 === 0) {
          console.log(`📈 Прогресс обучения: эпоха ${epoch}/${reducedEpochs}`);
        }
        
        this.simulatedAnnealing(reducedData);
        
        // Пауза для обновления UI
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    } else {
      console.log('⚡ Используем контрастивную дивергенцию');
      // Контрастивная дивергенция (оригинальный метод)
      const nSamples = data.length;
      const nBatches = Math.floor(nSamples / this.batchSize);
      console.log(`📊 Данные: ${nSamples} образцов, батчей: ${nBatches}, эпох: ${nEpochs}`);

      for (let epoch = 0; epoch < nEpochs; epoch++) {
        const indices = Array.from({ length: nSamples }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        for (let batchIdx = 0; batchIdx < nBatches; batchIdx++) {
          const batch: Float32Array[] = [];
          for (let i = 0; i < this.batchSize; i++) {
            const idx = indices[batchIdx * this.batchSize + i];
            batch.push(data[idx]);
          }
          this.contrastiveDivergence(batch);
          
          if (batchIdx % 5 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }

        if (progressCallback) {
          progressCallback(epoch + 1, nEpochs);
        }
        
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    }
  }

  reconstruct(sample: Float32Array): ReconstructionResult {
    const hidden = this.sampleHidden(sample);
    const reconstruction = this.sampleVisible(hidden);
    return { reconstruction, hidden };
  }

  saveToLocalStorage(): boolean {
    const data = {
      nVisible: this.nVisible,
      nHidden: this.nHidden,
      weights: this.weights.map(row => Array.from(row)),
      hiddenBias: Array.from(this.hiddenBias),
      visibleBias: Array.from(this.visibleBias),
      trainingMethod: this.trainingMethod,
      timestamp: Date.now()
    };
    
    try {
      const compressed = JSON.stringify(data);
      localStorage.setItem('rbm_weights', compressed);
      console.log('✅ Веса успешно сохранены в Local Storage');
      return true;
    } catch (e) {
      console.error('❌ Ошибка сохранения весов:', e);
      return false;
    }
  }

  static loadFromLocalStorage(): BernoulliRBM | null {
    try {
      const compressed = localStorage.getItem('rbm_weights');
      if (!compressed) return null;
      
      const data = JSON.parse(compressed);
      const rbm = new BernoulliRBM({
        nVisible: data.nVisible,
        nHidden: data.nHidden,
        learningRate: 0.06,
        batchSize: 32,
        trainingMethod: data.trainingMethod || 'contrastive-divergence'
      });
      
      rbm.weights = data.weights.map((row: number[]) => new Float32Array(row));
      rbm.hiddenBias = new Float32Array(data.hiddenBias);
      rbm.visibleBias = new Float32Array(data.visibleBias);
      
      console.log('✅ Веса успешно загружены из Local Storage');
      return rbm;
    } catch (e) {
      console.error('❌ Ошибка загрузки весов:', e);
      return null;
    }
  }

  getWeights(): Float32Array[] {
    return this.weights;
  }

  getHiddenBias(): Float32Array {
    return this.hiddenBias;
  }

  getVisibleBias(): Float32Array {
    return this.visibleBias;
  }

  getTrainingMethod(): TrainingMethod {
    return this.trainingMethod;
  }
}