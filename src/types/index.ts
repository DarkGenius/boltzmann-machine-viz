export type TrainingMethod = 'contrastive-divergence' | 'equilibrium';

export interface RBMParams {
  nVisible: number;
  nHidden: number;
  learningRate: number;
  batchSize: number;
  trainingMethod?: TrainingMethod;
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  progress: number;
  status: string;
}

export interface ReconstructionResult {
  reconstruction: Float32Array;
  hidden: Float32Array;
}

export interface ErrorMetrics {
  mse: string;
  errorPercent: string;
  pixelsDiff: string;
}

export type ViewMode = 'sample' | 'filters';

export interface TermDefinition {
  term: string;
  definition: string;
}

export interface MNISTSample {
  label: number;
  pixels: number[];
}

export type DataSource = 'generated' | 'mnist';