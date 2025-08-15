// Тестовый файл для проверки имитации отжига
// Этот файл можно запустить в браузере для отладки

// Простая реализация для тестирования
class TestRBM {
  constructor() {
    this.weights = [[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]];
    this.hiddenBias = [0.1, -0.1];
    this.visibleBias = [0.05, -0.05, 0.05];
    this.nVisible = 3;
    this.nHidden = 2;
    this.learningRate = 0.06;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  sampleHidden(visible) {
    const hidden = new Array(this.nHidden);
    for (let i = 0; i < this.nHidden; i++) {
      let activation = this.hiddenBias[i];
      for (let j = 0; j < this.nVisible; j++) {
        activation += visible[j] * this.weights[i][j];
      }
      hidden[i] = this.sigmoid(activation);
    }
    return hidden;
  }

  computeEnergy(visible, hidden) {
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

  simulatedAnnealing(data) {
    const initialTemperature = 1.0;
    const finalTemperature = 0.01;
    const coolingRate = 0.95;
    const iterationsPerTemperature = 10;
    
    let temperature = initialTemperature;
    let iterations = 0;
    
    console.log('Начинаем имитацию отжига...');
    
    while (temperature > finalTemperature) {
      for (let iter = 0; iter < iterationsPerTemperature; iter++) {
        // Выбираем случайный образец
        const sampleIndex = Math.floor(Math.random() * data.length);
        const sample = data[sampleIndex];
        
        // Получаем текущее состояние скрытого слоя
        const currentHidden = this.sampleHidden(sample);
        const currentEnergy = this.computeEnergy(sample, currentHidden);
        
        // Создаем новое состояние с небольшим случайным изменением
        const newHidden = new Array(this.nHidden);
        for (let i = 0; i < this.nHidden; i++) {
          const noise = (Math.random() - 0.5) * 0.1;
          newHidden[i] = Math.max(0, Math.min(1, currentHidden[i] + noise));
        }
        
        const newEnergy = this.computeEnergy(sample, newHidden);
        const energyDiff = newEnergy - currentEnergy;
        
        // Принимаем новое состояние с вероятностью, зависящей от температуры
        if (energyDiff < 0 || Math.random() < Math.exp(-energyDiff / temperature)) {
          // Обновляем веса в направлении снижения энергии
          for (let i = 0; i < this.nVisible; i++) {
            for (let j = 0; j < this.nHidden; j++) {
              const weightUpdate = this.learningRate * (sample[i] * newHidden[j] - sample[i] * currentHidden[j]);
              this.weights[j][i] += weightUpdate;
            }
          }
          
          // Обновляем смещения
          for (let j = 0; j < this.nHidden; j++) {
            this.hiddenBias[j] += this.learningRate * (newHidden[j] - currentHidden[j]);
          }
        }
        
        iterations++;
      }
      
      // Охлаждаем систему
      temperature *= coolingRate;
      
      if (iterations % 50 === 0) {
        console.log(`Итерация ${iterations}, температура: ${temperature.toFixed(4)}`);
      }
    }
    
    console.log(`Имитация отжига завершена за ${iterations} итераций`);
    console.log('Финальные веса:', this.weights);
    console.log('Финальные смещения скрытого слоя:', this.hiddenBias);
  }
}

// Тестовые данные
const testData = [
  [0.8, 0.2, 0.9],
  [0.1, 0.9, 0.3],
  [0.7, 0.4, 0.6]
];

// Создаем и тестируем RBM
const testRBM = new TestRBM();
console.log('Начальные веса:', testRBM.weights);
testRBM.simulatedAnnealing(testData);
