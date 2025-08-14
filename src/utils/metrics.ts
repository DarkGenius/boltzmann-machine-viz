import type { ErrorMetrics } from '../types';

export function calculateErrorMetrics(
  original: Float32Array,
  reconstruction: Float32Array
): ErrorMetrics {
  let mse = 0;
  let pixelsDifferent = 0;
  const threshold = 0.1;
  
  for (let i = 0; i < original.length; i++) {
    const diff = original[i] - reconstruction[i];
    mse += diff * diff;
    
    if (Math.abs(diff) > threshold) {
      pixelsDifferent++;
    }
  }
  
  mse = mse / original.length;
  const errorPercent = (Math.sqrt(mse) * 100).toFixed(2);
  const pixelsDiffPercent = ((pixelsDifferent / original.length) * 100).toFixed(1);
  
  return {
    mse: mse.toFixed(6),
    errorPercent: errorPercent + '%',
    pixelsDiff: `${pixelsDifferent}/784 (${pixelsDiffPercent}%)`
  };
}

export function getMSECategory(mse: number): 'excellent' | 'good' | 'poor' {
  if (mse < 0.01) return 'excellent';
  if (mse <= 0.05) return 'good';
  return 'poor';
}