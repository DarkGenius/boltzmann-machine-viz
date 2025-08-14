import type { MNISTSample } from '../types';

export async function loadRealMNIST(): Promise<Float32Array[]> {
  try {
    const response = await fetch('/data/mnist_2000.json');
    if (!response.ok) {
      throw new Error('Failed to load MNIST data');
    }
    
    const mnistData: MNISTSample[] = await response.json();
    
    // Конвертируем в Float32Array (данные уже нормализованы в [0, 1])
    const processedData = mnistData.map(sample => new Float32Array(sample.pixels));
    
    // Отладочная информация
    if (processedData.length > 0) {
      const firstSample = processedData[0];
      const nonZeroPixels = Array.from(firstSample).filter(p => p > 0).length;
      const maxValue = Math.max(...Array.from(firstSample));
      const minValue = Math.min(...Array.from(firstSample));
      
      console.log(`✅ Загружено ${processedData.length} реальных образцов MNIST`);
      console.log(`📊 Первый образец: ${nonZeroPixels}/784 ненулевых пикселей, диапазон [${minValue.toFixed(3)}, ${maxValue.toFixed(3)}]`);
    }
    
    return processedData;
  } catch (error) {
    console.error('❌ Ошибка загрузки MNIST данных:', error);
    throw new Error('Не удалось загрузить реальные данные MNIST. Используются сгенерированные данные.');
  }
}

export function getMNISTLabels(): Promise<number[]> {
  return fetch('/data/mnist_2000.json')
    .then(response => response.json())
    .then((data: MNISTSample[]) => data.map(sample => sample.label))
    .catch(() => {
      console.warn('Не удалось загрузить метки MNIST');
      return [];
    });
}