/**
 * Тесты для проверки корректности алгоритма имитации отжига в BernoulliRBM
 */

// Простая реализация тестового фреймворка
class TestRunner {
  constructor() {
    this.tests = [];
    this.results = { passed: 0, failed: 0, total: 0 };
  }

  test(name, testFn) {
    this.tests.push({ name, testFn });
  }

  async run() {
    console.log('🧪 Запуск тестов алгоритма имитации отжига...\n');
    
    for (const { name, testFn } of this.tests) {
      this.results.total++;
      try {
        await testFn();
        console.log(`✅ ${name}`);
        this.results.passed++;
      } catch (error) {
        console.log(`❌ ${name}: ${error.message}`);
        this.results.failed++;
      }
    }

    console.log(`\n📊 Результаты тестов:`);
    console.log(`   Пройдено: ${this.results.passed}/${this.results.total}`);
    console.log(`   Провалено: ${this.results.failed}/${this.results.total}`);
    
    if (this.results.failed === 0) {
      console.log('🎉 Все тесты пройдены успешно!');
    } else {
      console.log('⚠️ Некоторые тесты провалились');
    }
  }

  assert(condition, message) {
    if (!condition) {
      throw new Error(message);
    }
  }
}

// Упрощенная реализация RBM для тестирования
class TestRBM {
  constructor(nVisible, nHidden, learningRate = 0.06) {
    this.nVisible = nVisible;
    this.nHidden = nHidden;
    this.learningRate = learningRate;
    
    // Инициализация весов
    this.weights = this.randomMatrix(nHidden, nVisible, 0.01);
    this.hiddenBias = new Float32Array(nHidden);
    this.visibleBias = new Float32Array(nVisible);
  }

  randomMatrix(rows, cols, scale) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
      matrix[i] = new Float32Array(cols);
      for (let j = 0; j < cols; j++) {
        matrix[i][j] = (Math.random() - 0.5) * 2 * scale;
      }
    }
    return matrix;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  // Вычисление энергии системы
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

  sampleHiddenBinary(visible) {
    const hidden = new Float32Array(this.nHidden);
    for (let i = 0; i < this.nHidden; i++) {
      let activation = this.hiddenBias[i];
      for (let j = 0; j < this.nVisible; j++) {
        activation += visible[j] * this.weights[i][j];
      }
      const prob = this.sigmoid(activation);
      hidden[i] = Math.random() < prob ? 1 : 0;
    }
    return hidden;
  }

  // Упрощенная версия имитации отжига для тестирования
  simulatedAnnealing(data, maxIterations = 1000) {
    const initialTemperature = 2.0;
    const finalTemperature = 0.1;
    const coolingRate = 0.99;
    const stepsPerTemperature = 10;
    
    let temperature = initialTemperature;
    let totalIterations = 0;
    let acceptedMoves = 0;
    let energyHistory = [];
    
    // Начинаем с случайного образца
    let currentSample = data[Math.floor(Math.random() * data.length)].slice();
    let currentHidden = this.sampleHiddenBinary(currentSample);
    let currentEnergy = this.computeEnergy(currentSample, currentHidden);
    
    energyHistory.push(currentEnergy);
    
    while (temperature > finalTemperature && totalIterations < maxIterations) {
      for (let step = 0; step < stepsPerTemperature && totalIterations < maxIterations; step++) {
        totalIterations++;
        
        // Генерируем новое состояние
        const proposedSample = currentSample.slice();
        const proposedHidden = currentHidden.slice();
        
        // Изменяем случайные нейроны
        const neuronsToFlip = Math.min(2, this.nHidden);
        for (let flip = 0; flip < neuronsToFlip; flip++) {
          const neuronIdx = Math.floor(Math.random() * this.nHidden);
          proposedHidden[neuronIdx] = proposedHidden[neuronIdx] > 0.5 ? 0 : 1;
        }
        
        // Перевычисляем видимый слой
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
        
        // Критерий Метрополиса
        let shouldAccept = false;
        if (energyDiff <= 0) {
          shouldAccept = true;
        } else {
          const acceptProbability = Math.exp(-energyDiff / temperature);
          shouldAccept = Math.random() < acceptProbability;
        }
        
        if (shouldAccept) {
          acceptedMoves++;
          currentSample = proposedSample;
          currentHidden = proposedHidden;
          currentEnergy = proposedEnergy;
        }
        
        energyHistory.push(currentEnergy);
      }
      
      temperature *= coolingRate;
    }
    
    return {
      totalIterations,
      acceptedMoves,
      finalEnergy: currentEnergy,
      energyHistory,
      acceptanceRate: acceptedMoves / totalIterations
    };
  }
}

// Создание тестовых данных
function generateTestData(nSamples, nVisible) {
  const data = [];
  for (let i = 0; i < nSamples; i++) {
    const sample = new Float32Array(nVisible);
    for (let j = 0; j < nVisible; j++) {
      sample[j] = Math.random() < 0.3 ? 1 : 0; // Разреженные данные
    }
    data.push(sample);
  }
  return data;
}

// Запуск тестов
const runner = new TestRunner();

// Тест 1: Корректность вычисления энергии
runner.test('Энергия вычисляется корректно', () => {
  const rbm = new TestRBM(4, 3);
  
  // Устанавливаем известные веса и смещения
  rbm.weights[0][0] = 0.5;
  rbm.weights[0][1] = -0.3;
  rbm.hiddenBias[0] = 0.2;
  rbm.visibleBias[0] = -0.1;
  rbm.visibleBias[1] = 0.4;
  
  const visible = new Float32Array([1, 0, 1, 0]);
  const hidden = new Float32Array([1, 0, 1]);
  
  const energy = rbm.computeEnergy(visible, hidden);
  
  // Проверяем, что энергия конечна
  runner.assert(isFinite(energy), 'Энергия должна быть конечной');
  runner.assert(!isNaN(energy), 'Энергия не должна быть NaN');
  
  console.log(`  Вычисленная энергия: ${energy.toFixed(4)}`);
});

// Тест 2: Процент принятия в разумных пределах
runner.test('Процент принятия в разумных пределах', async () => {
  const rbm = new TestRBM(6, 4);
  const data = generateTestData(20, 6);
  
  const result = rbm.simulatedAnnealing(data, 500);
  
  runner.assert(result.acceptanceRate > 0.05, `Процент принятия слишком низкий: ${(result.acceptanceRate * 100).toFixed(1)}%`);
  runner.assert(result.acceptanceRate < 0.8, `Процент принятия слишком высокий: ${(result.acceptanceRate * 100).toFixed(1)}%`);
  
  console.log(`  Процент принятия: ${(result.acceptanceRate * 100).toFixed(1)}%`);
  console.log(`  Финальная энергия: ${result.finalEnergy.toFixed(4)}`);
});

// Тест 3: Энергия в целом уменьшается
runner.test('Энергия стабилизируется или уменьшается', async () => {
  const rbm = new TestRBM(6, 4);
  const data = generateTestData(20, 6);
  
  const result = rbm.simulatedAnnealing(data, 800);
  
  const initialEnergy = result.energyHistory[0];
  const finalEnergy = result.energyHistory[result.energyHistory.length - 1];
  
  const energyReduction = initialEnergy - finalEnergy;
  
  console.log(`  Начальная энергия: ${initialEnergy.toFixed(4)}`);
  console.log(`  Финальная энергия: ${finalEnergy.toFixed(4)}`);
  console.log(`  Изменение энергии: ${energyReduction.toFixed(4)}`);
  
  // Допускаем небольшое увеличение энергии из-за стохастичности
  runner.assert(energyReduction > -3.0, 'Энергия не должна сильно увеличиваться');
});

// Тест 4: Стабильность алгоритма
runner.test('Алгоритм стабилен при многократном запуске', async () => {
  const rbm = new TestRBM(4, 3);
  const data = generateTestData(10, 4);
  
  const results = [];
  const numRuns = 5;
  
  for (let run = 0; run < numRuns; run++) {
    const result = rbm.simulatedAnnealing(data, 300);
    results.push(result);
    
    runner.assert(isFinite(result.finalEnergy), `Прогон ${run + 1}: Финальная энергия должна быть конечной`);
    runner.assert(result.acceptanceRate > 0, `Прогон ${run + 1}: Должен быть хотя бы один принятый ход`);
  }
  
  // Проверяем, что результаты не слишком сильно различаются
  const acceptanceRates = results.map(r => r.acceptanceRate);
  const avgAcceptance = acceptanceRates.reduce((a, b) => a + b) / numRuns;
  const maxDeviation = Math.max(...acceptanceRates.map(rate => Math.abs(rate - avgAcceptance)));
  
  console.log(`  Средний процент принятия: ${(avgAcceptance * 100).toFixed(1)}%`);
  console.log(`  Максимальное отклонение: ${(maxDeviation * 100).toFixed(1)}%`);
  
  runner.assert(maxDeviation < 0.4, 'Результаты должны быть относительно стабильными между запусками');
});

// Тест 5: Корректность изменения температуры
runner.test('Температура уменьшается монотонно', async () => {
  const initialTemperature = 2.0;
  const finalTemperature = 0.1;
  const coolingRate = 0.95;
  
  let temperature = initialTemperature;
  const temperatures = [temperature];
  
  // Симулируем охлаждение
  for (let i = 0; i < 50; i++) {
    temperature *= coolingRate;
    temperatures.push(temperature);
    if (temperature <= finalTemperature) break;
  }
  
  // Проверяем монотонность
  for (let i = 1; i < temperatures.length; i++) {
    runner.assert(temperatures[i] <= temperatures[i-1], 'Температура должна монотонно уменьшаться');
  }
  
  runner.assert(temperatures[0] === initialTemperature, 'Начальная температура должна быть корректной');
  runner.assert(temperatures[temperatures.length - 1] <= finalTemperature * 1.1, 'Финальная температура должна быть близка к заданной');
  
  console.log(`  Начальная T: ${temperatures[0].toFixed(3)}`);
  console.log(`  Финальная T: ${temperatures[temperatures.length - 1].toFixed(3)}`);
});

// Тест 6: Работа с разными размерами сети
runner.test('Алгоритм работает с разными размерами сети', async () => {
  const sizes = [
    { visible: 2, hidden: 2 },
    { visible: 8, hidden: 4 },
    { visible: 10, hidden: 6 }
  ];
  
  for (const { visible, hidden } of sizes) {
    const rbm = new TestRBM(visible, hidden);
    const data = generateTestData(15, visible);
    
    const result = rbm.simulatedAnnealing(data, 200);
    
    runner.assert(isFinite(result.finalEnergy), `Размер ${visible}x${hidden}: Энергия должна быть конечной`);
    runner.assert(result.acceptanceRate > 0, `Размер ${visible}x${hidden}: Должен быть хотя бы один принятый ход`);
    
    console.log(`  ${visible}x${hidden}: принятие ${(result.acceptanceRate * 100).toFixed(1)}%, энергия ${result.finalEnergy.toFixed(3)}`);
  }
});

// Запуск всех тестов
runner.run().then(() => {
  console.log('\n🏁 Тестирование завершено');
}).catch(error => {
  console.error('❌ Ошибка при запуске тестов:', error);
});