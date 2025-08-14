import type { RBMParams, ReconstructionResult } from '../types';

export class BernoulliRBM {
  private nVisible: number;
  private nHidden: number;
  private learningRate: number;
  private batchSize: number;
  private weights: Float32Array[];
  private hiddenBias: Float32Array;
  private visibleBias: Float32Array;

  constructor({ nVisible, nHidden, learningRate = 0.06, batchSize = 32 }: RBMParams) {
    this.nVisible = nVisible;
    this.nHidden = nHidden;
    this.learningRate = learningRate;
    this.batchSize = batchSize;

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
    const nSamples = data.length;
    const nBatches = Math.floor(nSamples / this.batchSize);

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
        batchSize: 32
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
}