import { useState, useEffect, useRef } from 'react';
import { BernoulliRBM } from '../ml/BernoulliRBM';
import { drawImageColored } from '../utils/canvas';

interface NeuronAnalysisProps {
  rbm: BernoulliRBM;
  sample: Float32Array;
  hiddenActivations: Float32Array;
}

export function NeuronAnalysis({ rbm, sample, hiddenActivations }: NeuronAnalysisProps) {
  const [selectedNeuron, setSelectedNeuron] = useState(0);
  
  const filterCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const contributionCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    analyzeNeuron();
  }, [selectedNeuron, rbm, sample, hiddenActivations]);

  const analyzeNeuron = () => {
    const weights = rbm.getWeights();
    const activation = hiddenActivations[selectedNeuron];

    // 1. Фильтр нейрона (нормализованные веса)
    const filter = new Float32Array(784);
    let maxAbs = 0;
    for (let i = 0; i < 784; i++) {
      filter[i] = weights[selectedNeuron][i];
      maxAbs = Math.max(maxAbs, Math.abs(filter[i]));
    }
    if (maxAbs > 0) {
      for (let i = 0; i < 784; i++) {
        filter[i] = filter[i] / maxAbs;
      }
    }

    // 2. Наложение фильтра на образец
    const overlay = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      overlay[i] = sample[i] + filter[i] * activation * 0.3;
      overlay[i] = Math.max(-1, Math.min(1, overlay[i]));
    }

    // 3. Вклад нейрона в реконструкцию
    const contribution = new Float32Array(784);
    if (activation > 0.01) {
      for (let i = 0; i < 784; i++) {
        contribution[i] = Math.abs(weights[selectedNeuron][i] * activation);
      }
      
      let maxContrib = 0;
      for (let i = 0; i < 784; i++) {
        maxContrib = Math.max(maxContrib, contribution[i]);
      }
      if (maxContrib > 0) {
        for (let i = 0; i < 784; i++) {
          contribution[i] = contribution[i] / maxContrib;
        }
      }
    }

    // Отрисовка
    if (filterCanvasRef.current) {
      drawImageColored(filterCanvasRef.current, filter, 28, 28, 'bwr');
    }
    if (overlayCanvasRef.current) {
      drawImageColored(overlayCanvasRef.current, overlay, 28, 28, 'bwr');
    }
    if (contributionCanvasRef.current) {
      drawImageColored(contributionCanvasRef.current, contribution, 28, 28, 'hot');
    }
  };

  return (
    <div className="neuron-analysis">
      <div className="analysis-controls">
        <div className="slider-container">
          <div className="slider-label">
            Нейрон: <span>{selectedNeuron}</span>
            <span className="activation-badge">
              {hiddenActivations[selectedNeuron].toFixed(3)}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="63"
            value={selectedNeuron}
            onChange={(e) => setSelectedNeuron(parseInt(e.target.value))}
          />
        </div>
      </div>

      <div className="analysis-panels">
        <div className="analysis-panel">
          <div className="analysis-title">
            Фильтр нейрона
            <span className="help-icon">
              ?
              <div className="tooltip wide-tooltip">
                <div className="tooltip-content">
                  <div className="tooltip-header">Фильтр нейрона</div>
                  <div className="tooltip-text">
                    Чистые веса нейрона без модификаций<br />
                    Показывает, на какие паттерны "настроен" нейрон
                  </div>
                  <div className="tooltip-colors">
                    <div className="color-item">
                      <span style={{ color: '#6495ED' }}>Синий</span> = отрицательные веса
                    </div>
                    <div className="color-item">
                      <span style={{ color: '#FF6B6B' }}>Красный</span> = положительные веса
                    </div>
                  </div>
                </div>
              </div>
            </span>
          </div>
          <canvas ref={filterCanvasRef} width="168" height="168" />
        </div>

        <div className="analysis-panel">
          <div className="analysis-title">
            Наложение на образец
            <span className="help-icon">
              ?
              <div className="tooltip wide-tooltip">
                <div className="tooltip-content">
                  <div className="tooltip-header">Наложение на образец</div>
                  <div className="tooltip-text">
                    Комбинация: оригинал + фильтр × активация × 0.3<br />
                    Показывает, как фильтр "видит" конкретный образец<br />
                    Видны контуры цифры с наложенным паттерном<br />
                    Интенсивность зависит от активации нейрона
                  </div>
                </div>
              </div>
            </span>
          </div>
          <canvas ref={overlayCanvasRef} width="168" height="168" />
        </div>

        <div className="analysis-panel">
          <div className="analysis-title">
            Вклад в реконструкцию
            <span className="help-icon">
              ?
              <div className="tooltip wide-tooltip">
                <div className="tooltip-content">
                  <div className="tooltip-header">Вклад в реконструкцию</div>
                  <div className="tooltip-text">
                    Фактический вклад нейрона в финальную реконструкцию<br />
                    Черный = нейрон не активен (нет вклада)<br />
                    Яркие области = нейрон добавляет интенсивность<br />
                    Hot colormap: черный → красный → желтый → белый
                  </div>
                </div>
              </div>
            </span>
          </div>
          <canvas ref={contributionCanvasRef} width="168" height="168" />
        </div>
      </div>
    </div>
  );
}