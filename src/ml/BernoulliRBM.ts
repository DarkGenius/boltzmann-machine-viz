import type { RBMParams, ReconstructionResult, TrainingMethod } from '../types';

/**
 * Бернуллиевская машина Больцмана (RBM) - неориентированная вероятностная модель
 * для обучения представлений данных без учителя
 */
export class BernoulliRBM {
  private nVisible: number;
  private nHidden: number;
  private learningRate: number;
  private batchSize: number;
  private trainingMethod: TrainingMethod;
  private weights: Float32Array[];
  private hiddenBias: Float32Array;
  private visibleBias: Float32Array;

  /**
   * Создает новый экземпляр машины Больцмана
   * @param params - параметры инициализации RBM
   * @param params.nVisible - количество видимых нейронов
   * @param params.nHidden - количество скрытых нейронов
   * @param params.learningRate - скорость обучения (по умолчанию 0.06)
   * @param params.batchSize - размер батча (по умолчанию 32)
   * @param params.trainingMethod - метод обучения (по умолчанию 'contrastive-divergence')
   */
  constructor({ nVisible, nHidden, learningRate = 0.06, batchSize = 32, trainingMethod = 'contrastive-divergence' }: RBMParams) {
    this.nVisible = nVisible;
    this.nHidden = nHidden;
    this.learningRate = learningRate;
    this.batchSize = batchSize;
    this.trainingMethod = trainingMethod;

    const weightsScale = trainingMethod === 'equilibrium' ? 0.005 : 0.01;
    this.weights = this.randomMatrix(nHidden, nVisible, weightsScale);
    this.hiddenBias = new Float32Array(nHidden);
    this.visibleBias = new Float32Array(nVisible);
  }

  /**
   * Создает случайную матрицу весов с заданным масштабом
   * @param rows - количество строк
   * @param cols - количество столбцов
   * @param scale - масштаб случайных значений
   * @returns матрица случайных весов
   */
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

  /**
   * Сигмоидная функция активации
   * @param x - входное значение
   * @returns значение сигмоидной функции от 0 до 1
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * Вычисляет энергию системы (функция Ляпунова) для заданного состояния
   * @param visible - активации видимого слоя
   * @param hidden - активации скрытого слоя
   * @returns энергия системы
   */
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
   * Базовый метод для вычисления вероятностей активации нейронов
   * @param input - входные активации
   * @param bias - смещения для выходного слоя
   * @param weights - веса между слоями
   * @param outputSize - размер выходного слоя
   * @param inputSize - размер входного слоя
   * @param isTransposed - флаг для транспонирования матрицы весов
   * @returns вероятности активации выходного слоя
   */
  private sampleBaseProb(
    input: Float32Array,
    bias: Float32Array,
    weights: Float32Array[],
    outputSize: number,
    inputSize: number,
    isTransposed: boolean = false
  ): Float32Array {
    const output = new Float32Array(outputSize);
    for (let i = 0; i < outputSize; i++) {
      let activation = bias[i];
      for (let j = 0; j < inputSize; j++) {
        const weight = isTransposed ? weights[j][i] : weights[i][j];
        activation += input[j] * weight;
      }
      output[i] = this.sigmoid(activation);
    }
    return output;
  }

  /**
   * Базовый метод для бинарной выборки нейронов
   * @param input - входные активации
   * @param bias - смещения для выходного слоя
   * @param weights - веса между слоями
   * @param outputSize - размер выходного слоя
   * @param inputSize - размер входного слоя
   * @param isTransposed - флаг для транспонирования матрицы весов
   * @returns бинарные активации выходного слоя
   */
  private sampleBase(
    input: Float32Array,
    bias: Float32Array,
    weights: Float32Array[],
    outputSize: number,
    inputSize: number,
    isTransposed: boolean = false
  ): Float32Array {
    const probs = this.sampleBaseProb(input, bias, weights, outputSize, inputSize, isTransposed);
    const output = new Float32Array(outputSize);
    for (let i = 0; i < outputSize; i++) {
      // Бинарная выборка: 1 с вероятностью prob, 0 иначе
      output[i] = Math.random() < probs[i] ? 1 : 0;
    }
    return output;
  }

  /**
   * Вычисление вероятностей активации скрытого слоя
   */
  private sampleHidden(visible: Float32Array): Float32Array {
    return this.sampleBaseProb(visible, this.hiddenBias, this.weights, this.nHidden, this.nVisible, false);
  }

  /**
   * Вычисление вероятностей активации видимого слоя
   */
  private sampleVisible(hidden: Float32Array): Float32Array {
    return this.sampleBaseProb(hidden, this.visibleBias, this.weights, this.nVisible, this.nHidden, true);
  }

  /**
   * Бинарная выборка скрытого слоя (используется для выборки Гиббса)
   * Возвращает бинарные активации вместо вероятностей
   */
  private sampleHiddenBinary(visible: Float32Array): Float32Array {
    return this.sampleBase(visible, this.hiddenBias, this.weights, this.nHidden, this.nVisible, false);
  }

  /**
   * Бинарная выборка видимого слоя (используется для выборки Гиббса)
   * Возвращает бинарные активации вместо вероятностей
   */
  private sampleVisibleBinary(hidden: Float32Array): Float32Array {
    return this.sampleBase(hidden, this.visibleBias, this.weights, this.nVisible, this.nHidden, true);
  }

  /**
   * Создает нулевую матрицу заданного размера
   * @param rows - количество строк
   * @param cols - количество столбцов
   * @returns нулевая матрица
   */
  private zeroMatrix(rows: number, cols: number): Float32Array[] {
    return Array(rows).fill(null).map(() => new Float32Array(cols));
  }



  /**
   * Создает перемешанный массив индексов для случайного порядка данных
   * @param length - длина массива индексов
   * @returns перемешанный массив индексов от 0 до length-1
   */
  private getShuffledIndices(length: number): number[] {
    const indices = Array.from({ length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    return indices;
  }

  private gibbsSampleWithEnergy(visible: Float32Array, steps: number = 1): { visible: Float32Array, hidden: Float32Array, energy: number[] } {
    let v = visible.slice();
    let h = this.sampleHiddenBinary(v);
    const energies: number[] = [];

    for (let step = 0; step < steps; step++) {
      h = this.sampleHiddenBinary(v);
      v = this.sampleVisibleBinary(h);
      energies.push(this.computeEnergy(v, h));
    }

    return { visible: v, hidden: h, energy: energies };
  }

  /**
   * Обучение методом сэмплирования из равновесия
   * Использует длительное сэмплирование Гиббса для достижения равновесия
   * @param batch - батч обучающих данных
   * @param gibbsSteps - количество шагов Гиббса для достижения равновесия (по умолчанию 2000)
   * @param negPhaseSamples - количество сэмплов для отрицательной фазы (по умолчанию 500)
   */
  private equilibriumLearning(batch: Float32Array[], gibbsSteps: number = 3000, negPhaseSamples: number = 500): void {
    const batchSize = batch.length;
    console.log(`Начинаем сэмплирование из равновесия (batchSize = ${batchSize})...`);

    // Используем меньшую скорость обучения для equilibrium
    const effectiveLearningRate = 0.01; // this.learningRate * 0.1;

    // Положительная фаза: среднее по данным
    let posPhase = this.zeroMatrix(this.nHidden, this.nVisible);
    let posHidden = new Float32Array(this.nHidden);
    let posVisible = new Float32Array(this.nVisible);

    for (const sample of batch) {
      const hProb = this.sampleHidden(sample); // P(h|v),
      for (let i = 0; i < this.nHidden; i++) {
        for (let j = 0; j < this.nVisible; j++) {
          posPhase[i][j] += hProb[i] * sample[j];
        }
        posHidden[i] += hProb[i];
      }
      for (let j = 0; j < this.nVisible; j++) {
        posVisible[j] += sample[j];
      }
    }

    // Усредняем
    const scale = 1 / batchSize;
    for (let i = 0; i < this.nHidden; i++) {
      for (let j = 0; j < this.nVisible; j++) {
        posPhase[i][j] *= scale;
      }
      posHidden[i] *= scale;
    }
    for (let j = 0; j < this.nVisible; j++) {
      posVisible[j] *= scale;
    }

    // Отрицательная фаза: сэмплирование из модели
    const negPhase = this.zeroMatrix(this.nHidden, this.nVisible);
    const negHidden = new Float32Array(this.nHidden);
    const negVisible = new Float32Array(this.nVisible);

    // Начальное состояние (случайное)
    // let v = this.randomBinaryVector(this.nVisible);
    let v = batch[0].slice();
    let h = this.sampleHiddenBinary(v);

    // Burn-in: 2000 (по умолчанию) шагов Gibbs
    console.log(`🔄 Начальная энергия: ${this.computeEnergy(v, h).toFixed(1)}`);
    for (let i = 0; i < gibbsSteps; i++) {
      h = this.sampleHiddenBinary(v);
      v = this.sampleVisibleBinary(h);

      if (i % 500 === 0) {
        const energy = this.computeEnergy(v, h);
        console.log(`⚡ Шаг ${i}: энергия = ${energy.toFixed(1)}`);
      }
    }

    const { energy } = this.gibbsSampleWithEnergy(v, 100);
    console.log(`⚡ Энергия после burn-in: ${energy.slice(-10).map(e => e.toFixed(1)).join(', ')}`);

    // Собираем 500 (по умолчанию) сэмплов
    for (let s = 0; s < negPhaseSamples; s++) {
      // Используем бинарную выборку для перехода между состояниями
      h = this.sampleHiddenBinary(v);
      v = this.sampleVisibleBinary(h);

      for (let i = 0; i < this.nHidden; i++) {
        for (let j = 0; j < this.nVisible; j++) {
          negPhase[i][j] += h[i] * v[j];
        }
        negHidden[i] += h[i];
      }
      for (let j = 0; j < this.nVisible; j++) {
        negVisible[j] += v[j];
      }
    }

    // Усредняем
    const negScale = 1 / negPhaseSamples;
    for (let i = 0; i < this.nHidden; i++) {
      for (let j = 0; j < this.nVisible; j++) {
        negPhase[i][j] *= negScale;
      }
      negHidden[i] *= negScale;
    }
    for (let j = 0; j < this.nVisible; j++) {
      negVisible[j] *= negScale;
    }

    const avgHiddenActivation = negHidden.reduce((a, b) => a + b, 0) / this.nHidden;
    console.log(`📉 Средняя активация скрытых нейронов (model): ${avgHiddenActivation.toFixed(3)}`);

    // Обновляем веса
    for (let i = 0; i < this.nHidden; i++) {
      for (let j = 0; j < this.nVisible; j++) {
        this.weights[i][j] += effectiveLearningRate * (posPhase[i][j] - negPhase[i][j]);
        // L2 регуляризация
        this.weights[i][j] *= 0.99; // лёгкое затухание
      }
      this.hiddenBias[i] += effectiveLearningRate * (posHidden[i] - negHidden[i]);
    }
    for (let j = 0; j < this.nVisible; j++) {
      this.visibleBias[j] += effectiveLearningRate * (posVisible[j] - negVisible[j]);
    }
  }

  /**
   * Обучение методом контрастивной дивергенции (CD-1)
   * Быстрый приближенный алгоритм обучения RBM
   * @param batch - батч обучающих данных
   */
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

  /**
   * Обучает RBM на предоставленных данных
   * @param data - массив обучающих образцов
   * @param nEpochs - количество эпох обучения (по умолчанию 15)
   * @param progressCallback - функция обратного вызова для отслеживания прогресса
   * @returns Promise, который разрешается по завершении обучения
   */
  async fit(
    data: Float32Array[],
    nEpochs = 15,
    progressCallback?: (epoch: number, totalEpochs: number) => void
  ): Promise<void> {
    console.log(`🚀 Начинаем обучение методом: ${this.trainingMethod}`);

    if (this.trainingMethod === 'equilibrium') {
      console.log('❄️ Используем сэмплирование из равновесия');
      // Для сэмплирования из равновесия используем меньше данных
      const reducedData = data.slice(0, Math.min(20, data.length));
      const reducedEpochs = Math.min(10, nEpochs);
      const nBatches = Math.ceil(reducedData.length / this.batchSize);
      const realBatchSize = Math.min(this.batchSize, reducedData.length);
      console.log(`📊 Данные: ${reducedData.length} образцов, батчей: ${nBatches}, эпох: ${reducedEpochs}`);

      for (let epoch = 0; epoch < reducedEpochs; epoch++) {
        console.log(`📈 Прогресс обучения: эпоха ${epoch + 1}/${reducedEpochs}`);

        const indices = this.getShuffledIndices(reducedData.length);

        for (let batchIdx = 0; batchIdx < nBatches; batchIdx++) {
          const batch: Float32Array[] = [];
          for (let i = 0; i < realBatchSize; i++) {
            const idx = indices[batchIdx * realBatchSize + i];
            batch.push(data[idx]);
          }
          this.equilibriumLearning(batch);

          if (batchIdx % 5 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }

        if (progressCallback) {
          progressCallback(epoch + 1, reducedEpochs);
        }
        // Пауза для обновления UI
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      const sample = reducedData[0];
      const hiddenProbs = this.sampleHidden(sample);
      console.log('Активации скрытых нейронов:', Array.from(hiddenProbs));
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

  /**
   * Реконструирует входные данные через скрытое представление
   * @param sample - входной образец для реконструкции
   * @returns объект с реконструированными данными и скрытым представлением
   */
  reconstruct(sample: Float32Array): ReconstructionResult {
    const hidden = this.sampleHidden(sample);
    const reconstruction = this.sampleVisible(hidden);
    return { reconstruction, hidden };
  }

  /**
   * Сохраняет обученные веса в Local Storage браузера
   * @returns true если сохранение прошло успешно, false в противном случае
   */
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

  /**
   * Загружает обученные веса из Local Storage браузера
   * @returns экземпляр BernoulliRBM с загруженными весами или null при ошибке
   */
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

  /**
   * Возвращает матрицу весов между слоями
   * @returns двумерный массив весов [скрытые][видимые]
   */
  getWeights(): Float32Array[] {
    return this.weights;
  }

  /**
   * Возвращает смещения скрытого слоя
   * @returns массив смещений скрытых нейронов
   */
  getHiddenBias(): Float32Array {
    return this.hiddenBias;
  }

  /**
   * Возвращает смещения видимого слоя
   * @returns массив смещений видимых нейронов
   */
  getVisibleBias(): Float32Array {
    return this.visibleBias;
  }

  /**
   * Возвращает используемый метод обучения
   * @returns текущий метод обучения
   */
  getTrainingMethod(): TrainingMethod {
    return this.trainingMethod;
  }
}