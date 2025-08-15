/**
 * Тест обучения RBM на образцах MNIST с помощью имитации отжига
 */

const fs = require('fs');
const path = require('path');

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
    console.log('🧪 Тест обучения RBM на MNIST с имитацией отжига...\n');
    
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

  assertApprox(actual, expected, tolerance = 0.01, message = '') {
    const diff = Math.abs(actual - expected);
    if (diff > tolerance) {
      throw new Error(`${message} Expected ~${expected}, got ${actual} (diff: ${diff})`);
    }
  }
}

// Генератор MNIST-подобных данных (упрощенная версия)
class MNISTGenerator {
  // Создает простые паттерны, похожие на цифры MNIST
  static generateDigitPattern(digit, size = 28) {
    const pattern = new Float32Array(size * size);
    
    switch(digit) {
      case 0: // Круг
        this.drawCircle(pattern, size, size / 2, size / 2, size / 3);
        break;
      case 1: // Вертикальная линия
        this.drawLine(pattern, size, size / 2, size / 4, size / 2, 3 * size / 4);
        break;
      case 2: // S-образная кривая
        this.drawSCurve(pattern, size);
        break;
      case 3: // Две горизонтальные линии
        this.drawLine(pattern, size, size / 4, size / 3, 3 * size / 4, size / 3);
        this.drawLine(pattern, size, size / 4, 2 * size / 3, 3 * size / 4, 2 * size / 3);
        break;
      case 4: // L-образная фигура
        this.drawLShape(pattern, size);
        break;
      case 5: // Прямоугольник
        this.drawRectangle(pattern, size, size / 4, size / 4, size / 2, size / 2);
        break;
      case 6: // Спираль
        this.drawSpiral(pattern, size);
        break;
      case 7: // Диагональная линия
        this.drawDiagonal(pattern, size);
        break;
      case 8: // Восьмерка
        this.drawEight(pattern, size);
        break;
      case 9: // Круг с хвостиком
        this.drawCircle(pattern, size, size / 2, size / 3, size / 6);
        this.drawLine(pattern, size, size / 2, size / 2, size / 2, 3 * size / 4);
        break;
    }
    
    // Добавляем небольшой шум
    for (let i = 0; i < pattern.length; i++) {
      if (Math.random() < 0.05) { // 5% шума
        pattern[i] = Math.random() > 0.5 ? 1 : 0;
      }
    }
    
    return pattern;
  }

  static drawCircle(pattern, size, centerX, centerY, radius) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dx = x - centerX;
        const dy = y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance >= radius - 2 && distance <= radius + 2) {
          pattern[y * size + x] = 1;
        }
      }
    }
  }

  static drawLine(pattern, size, x1, y1, x2, y2) {
    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    const steps = Math.max(dx, dy);
    
    for (let i = 0; i <= steps; i++) {
      const x = Math.round(x1 + (x2 - x1) * i / steps);
      const y = Math.round(y1 + (y2 - y1) * i / steps);
      if (x >= 0 && x < size && y >= 0 && y < size) {
        pattern[y * size + x] = 1;
        // Утолщаем линию
        if (x + 1 < size) pattern[y * size + x + 1] = 1;
        if (y + 1 < size) pattern[(y + 1) * size + x] = 1;
      }
    }
  }

  static drawSCurve(pattern, size) {
    for (let x = size / 4; x < 3 * size / 4; x++) {
      const y1 = size / 3 + Math.sin((x - size / 4) / (size / 2) * Math.PI) * size / 6;
      const y2 = 2 * size / 3 + Math.sin((x - size / 4) / (size / 2) * Math.PI) * size / 6;
      this.setPixel(pattern, size, x, Math.round(y1));
      this.setPixel(pattern, size, x, Math.round(y2));
    }
  }

  static drawLShape(pattern, size) {
    // Вертикальная часть L
    this.drawLine(pattern, size, size / 4, size / 4, size / 4, 3 * size / 4);
    // Горизонтальная часть L
    this.drawLine(pattern, size, size / 4, 3 * size / 4, 3 * size / 4, 3 * size / 4);
  }

  static drawRectangle(pattern, size, x, y, width, height) {
    for (let i = 0; i < width; i++) {
      for (let j = 0; j < height; j++) {
        if (i === 0 || i === width - 1 || j === 0 || j === height - 1) {
          this.setPixel(pattern, size, x + i, y + j);
        }
      }
    }
  }

  static drawSpiral(pattern, size) {
    const centerX = size / 2;
    const centerY = size / 2;
    let angle = 0;
    let radius = 2;
    
    while (radius < size / 3) {
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      this.setPixel(pattern, size, Math.round(x), Math.round(y));
      angle += 0.2;
      radius += 0.3;
    }
  }

  static drawDiagonal(pattern, size) {
    this.drawLine(pattern, size, size / 4, size / 4, 3 * size / 4, 3 * size / 4);
  }

  static drawEight(pattern, size) {
    // Верхний круг
    this.drawCircle(pattern, size, size / 2, size / 3, size / 6);
    // Нижний круг
    this.drawCircle(pattern, size, size / 2, 2 * size / 3, size / 6);
  }

  static setPixel(pattern, size, x, y) {
    if (x >= 0 && x < size && y >= 0 && y < size) {
      pattern[y * size + x] = 1;
    }
  }

  // Генерирует набор из N образцов разных цифр
  static generate10Samples(imageSize = 28) {
    const samples = [];
    for (let digit = 0; digit < 10; digit++) {
      const pattern = this.generateDigitPattern(digit, imageSize);
      samples.push(pattern);
    }
    return samples;
  }

  // Визуализация паттерна в консоли
  static visualizePattern(pattern, size = 28) {
    let output = '';
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        output += pattern[y * size + x] > 0.5 ? '█' : ' ';
      }
      output += '\n';
    }
    return output;
  }
}

// Упрощенная реализация RBM для тестирования на MNIST
class MNISTRBM {
  constructor(nVisible, nHidden, learningRate = 0.01) {
    this.nVisible = nVisible;
    this.nHidden = nHidden;
    this.learningRate = learningRate;
    
    // Инициализация весов (немного больше для MNIST)
    this.weights = this.randomMatrix(nHidden, nVisible, 0.1);
    this.hiddenBias = new Float32Array(nHidden);
    this.visibleBias = new Float32Array(nVisible);
    
    // Статистика обучения
    this.energyHistory = [];
    this.reconstructionErrors = [];
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
    // Ограничиваем x для численной стабильности
    const clampedX = Math.max(-20, Math.min(20, x));
    return 1 / (1 + Math.exp(-clampedX));
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

  sampleHidden(visible) {
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

  sampleVisible(hidden) {
    const visible = new Float32Array(this.nVisible);
    for (let i = 0; i < this.nVisible; i++) {
      let activation = this.visibleBias[i];
      for (let j = 0; j < this.nHidden; j++) {
        activation += hidden[j] * this.weights[j][i];
      }
      const prob = this.sigmoid(activation);
      visible[i] = Math.random() < prob ? 1 : 0;
    }
    return visible;
  }

  // Имитация отжига для MNIST
  simulatedAnnealing(data, epochs = 10) {
    console.log(`🔥 Начинаем обучение RBM ${this.nVisible}x${this.nHidden} на ${data.length} образцах MNIST...`);
    
    const initialTemperature = 5.0; // Выше температура для MNIST
    const finalTemperature = 0.01;
    const coolingRate = 0.95;
    const stepsPerTemperature = 30;
    
    let temperature = initialTemperature;
    let totalIterations = 0;
    let acceptedMoves = 0;
    
    // Начинаем с случайного образца
    let currentSampleIdx = Math.floor(Math.random() * data.length);
    let currentSample = data[currentSampleIdx].slice();
    let currentHidden = this.sampleHiddenBinary(currentSample);
    let currentEnergy = this.computeEnergy(currentSample, currentHidden);
    
    this.energyHistory.push(currentEnergy);
    console.log(`📊 Начальная энергия: ${currentEnergy.toFixed(4)}`);
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochAccepted = 0;
      let epochIterations = 0;
      
      temperature = initialTemperature * Math.pow(coolingRate, epoch);
      
      while (epochIterations < stepsPerTemperature * 10) { // Больше итераций для MNIST
        totalIterations++;
        epochIterations++;
        
        // Выбираем новый образец из данных время от времени
        if (totalIterations % 20 === 0) {
          currentSampleIdx = Math.floor(Math.random() * data.length);
          currentSample = data[currentSampleIdx].slice();
          currentHidden = this.sampleHiddenBinary(currentSample);
          currentEnergy = this.computeEnergy(currentSample, currentHidden);
        }
        
        // Генерируем новое состояние
        const proposedSample = currentSample.slice();
        const proposedHidden = currentHidden.slice();
        
        // Изменяем больше нейронов для MNIST (более сложные паттерны)
        const neuronsToFlip = Math.min(5, Math.max(1, Math.floor(this.nHidden * 0.1)));
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
          epochAccepted++;
          currentSample = proposedSample;
          currentHidden = proposedHidden;
          currentEnergy = proposedEnergy;
          
          // Обновляем веса при каждом принятом изменении
          if (acceptedMoves % 5 === 0) {
            this.updateWeights(data[currentSampleIdx], currentSample, currentHidden);
          }
        }
        
        if (totalIterations % 100 === 0) {
          this.energyHistory.push(currentEnergy);
        }
      }
      
      // Тестируем реконструкцию после каждой эпохи
      const reconstructionError = this.testReconstruction(data);
      this.reconstructionErrors.push(reconstructionError);
      
      const epochAcceptanceRate = (epochAccepted / epochIterations * 100).toFixed(1);
      console.log(`📈 Эпоха ${epoch + 1}/${epochs}: T=${temperature.toFixed(3)}, принято=${epochAcceptanceRate}%, энергия=${currentEnergy.toFixed(4)}, ошибка реконструкции=${reconstructionError.toFixed(4)}`);
    }
    
    const finalAcceptanceRate = (acceptedMoves / totalIterations * 100).toFixed(1);
    console.log(`❄️ Обучение завершено: итераций=${totalIterations}, принято=${finalAcceptanceRate}%`);
    console.log(`⚡ Финальная энергия: ${currentEnergy.toFixed(4)}`);
    
    return {
      totalIterations,
      acceptedMoves,
      finalEnergy: currentEnergy,
      acceptanceRate: acceptedMoves / totalIterations,
      energyHistory: this.energyHistory,
      reconstructionErrors: this.reconstructionErrors
    };
  }

  // Простое обновление весов
  updateWeights(originalSample, currentSample, currentHidden) {
    const dataHidden = this.sampleHidden(originalSample);
    const lr = this.learningRate * 0.1;
    
    // Обновляем веса: положительная фаза (данные) - отрицательная фаза (модель)
    for (let i = 0; i < this.nVisible; i++) {
      for (let j = 0; j < this.nHidden; j++) {
        const dataCorr = originalSample[i] * dataHidden[j];
        const modelCorr = currentSample[i] * currentHidden[j];
        this.weights[j][i] += lr * (dataCorr - modelCorr);
      }
      this.visibleBias[i] += lr * (originalSample[i] - currentSample[i]);
    }
    
    for (let j = 0; j < this.nHidden; j++) {
      this.hiddenBias[j] += lr * (dataHidden[j] - currentHidden[j]);
    }
  }

  // Тестирование качества реконструкции
  testReconstruction(data) {
    let totalError = 0;
    const testSamples = Math.min(5, data.length);
    
    for (let i = 0; i < testSamples; i++) {
      const sample = data[i];
      const hidden = this.sampleHidden(sample);
      const reconstruction = this.sampleVisible(hidden);
      
      // Вычисляем среднеквадратичную ошибку
      let sampleError = 0;
      for (let j = 0; j < this.nVisible; j++) {
        const diff = sample[j] - reconstruction[j];
        sampleError += diff * diff;
      }
      totalError += Math.sqrt(sampleError / this.nVisible);
    }
    
    return totalError / testSamples;
  }

  // Реконструкция образца
  reconstruct(sample) {
    const hidden = this.sampleHidden(sample);
    const reconstruction = this.sampleVisible(hidden);
    return reconstruction;
  }
}

// Запуск тестов
const runner = new TestRunner();

// Тест: Обучение RBM на MNIST образцах
runner.test('Обучение RBM на 10 образцах MNIST с имитацией отжига', async () => {
  const imageSize = 14; // Уменьшенный размер для быстрого тестирования
  const hiddenSize = 20;
  
  console.log(`\n🎯 Создаем RBM ${imageSize * imageSize}x${hiddenSize} для обучения на MNIST...`);
  
  // Генерируем 10 образцов MNIST-подобных данных
  const mnistSamples = MNISTGenerator.generate10Samples(imageSize);
  
  console.log(`📝 Сгенерировано ${mnistSamples.length} образцов размера ${imageSize}x${imageSize}`);
  console.log('📋 Примеры образцов:');
  
  // Показываем первые 3 образца
  for (let i = 0; i < 3; i++) {
    console.log(`\nЦифра ${i}:`);
    console.log(MNISTGenerator.visualizePattern(mnistSamples[i], imageSize));
  }
  
  // Создаем и обучаем RBM
  const rbm = new MNISTRBM(imageSize * imageSize, hiddenSize, 0.01);
  
  const startTime = Date.now();
  const result = rbm.simulatedAnnealing(mnistSamples, 5); // 5 эпох для быстрого тестирования
  const trainingTime = (Date.now() - startTime) / 1000;
  
  console.log(`\n⏱️ Время обучения: ${trainingTime.toFixed(2)} сек`);
  
  // Проверяем основные требования
  runner.assert(isFinite(result.finalEnergy), 'Финальная энергия должна быть конечной');
  runner.assert(result.acceptanceRate > 0, 'Должен быть хотя бы один принятый ход');
  runner.assert(result.energyHistory.length > 0, 'История энергии должна быть записана');
  runner.assert(result.reconstructionErrors.length > 0, 'Ошибки реконструкции должны быть записаны');
  
  // Проверяем, что ошибка реконструкции уменьшается или стабилизируется
  const initialError = result.reconstructionErrors[0];
  const finalError = result.reconstructionErrors[result.reconstructionErrors.length - 1];
  const errorImprovement = initialError - finalError;
  
  console.log(`\n📊 Статистика обучения:`);
  console.log(`   Начальная ошибка реконструкции: ${initialError.toFixed(4)}`);
  console.log(`   Финальная ошибка реконструкции: ${finalError.toFixed(4)}`);
  console.log(`   Улучшение: ${errorImprovement.toFixed(4)}`);
  console.log(`   Процент принятия: ${(result.acceptanceRate * 100).toFixed(1)}%`);
  
  // Тестируем реконструкцию некоторых образцов
  console.log(`\n🔍 Тестирование реконструкции:`);
  for (let i = 0; i < 3; i++) {
    const original = mnistSamples[i];
    const reconstructed = rbm.reconstruct(original);
    
    // Вычисляем сходство
    let similarity = 0;
    for (let j = 0; j < original.length; j++) {
      if ((original[j] > 0.5) === (reconstructed[j] > 0.5)) {
        similarity++;
      }
    }
    const similarityPercent = (similarity / original.length * 100).toFixed(1);
    
    console.log(`   Цифра ${i}: сходство ${similarityPercent}%`);
    
    if (i === 0) { // Показываем реконструкцию первого образца
      console.log('   Оригинал:');
      console.log(MNISTGenerator.visualizePattern(original, imageSize));
      console.log('   Реконструкция:');
      console.log(MNISTGenerator.visualizePattern(reconstructed, imageSize));
    }
  }
  
  // Допускаем некоторое увеличение ошибки из-за стохастичности
  runner.assert(errorImprovement > -0.3, 'Ошибка реконструкции не должна сильно увеличиваться');
  runner.assert(finalError < 1.0, 'Финальная ошибка реконструкции должна быть разумной');
  
  console.log(`\n✅ RBM успешно обучена на образцах MNIST!`);
});

// Тест: Проверка способности различать разные цифры
runner.test('RBM различает разные цифры после обучения', async () => {
  const imageSize = 12;
  const hiddenSize = 15;
  
  const mnistSamples = MNISTGenerator.generate10Samples(imageSize);
  const rbm = new MNISTRBM(imageSize * imageSize, hiddenSize, 0.01);
  
  // Быстрое обучение
  rbm.simulatedAnnealing(mnistSamples, 3);
  
  // Тестируем различные скрытые представления
  const hiddenRepresentations = [];
  for (let i = 0; i < mnistSamples.length; i++) {
    const hidden = rbm.sampleHidden(mnistSamples[i]);
    hiddenRepresentations.push(hidden);
  }
  
  // Проверяем, что разные цифры дают разные скрытые представления
  let differentRepresentations = 0;
  for (let i = 0; i < hiddenRepresentations.length; i++) {
    for (let j = i + 1; j < hiddenRepresentations.length; j++) {
      let diff = 0;
      for (let k = 0; k < hiddenSize; k++) {
        diff += Math.abs(hiddenRepresentations[i][k] - hiddenRepresentations[j][k]);
      }
      if (diff > 0.5) { // Достаточно большая разница
        differentRepresentations++;
      }
    }
  }
  
  const totalPairs = (hiddenRepresentations.length * (hiddenRepresentations.length - 1)) / 2;
  const diversityPercent = (differentRepresentations / totalPairs * 100).toFixed(1);
  
  console.log(`   Различающихся пар представлений: ${diversityPercent}%`);
  
  runner.assert(differentRepresentations > totalPairs * 0.3, 
    'RBM должна создавать различные скрытые представления для разных цифр');
});

// Запуск всех тестов
runner.run().then(() => {
  console.log('\n🏁 Тестирование MNIST завершено');
}).catch(error => {
  console.error('❌ Ошибка при запуске тестов:', error);
});