import { useEffect, useRef, useState } from 'react';
import { BernoulliRBM } from '../ml/BernoulliRBM';
import { drawImage } from '../utils/canvas';
import { calculateErrorMetrics, getMSECategory } from '../utils/metrics';
import { ErrorStats } from './ErrorStats';
import { NeuronAnalysis } from './NeuronAnalysis';

interface SampleVisualizationProps {
  rbm: BernoulliRBM;
  data: Float32Array[];
}

export function SampleVisualization({ rbm, data }: SampleVisualizationProps) {
  const [sampleIndex, setSampleIndex] = useState(0);
  const [currentSample, setCurrentSample] = useState<Float32Array | null>(null);
  const [reconstruction, setReconstruction] = useState<Float32Array | null>(null);
  const [hiddenActivations, setHiddenActivations] = useState<Float32Array | null>(null);
  
  const originalCanvasRef = useRef<HTMLCanvasElement>(null);
  const reconstructionCanvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!rbm || !data) return;

    const sample = data[sampleIndex];
    const result = rbm.reconstruct(sample);
    
    setCurrentSample(sample);
    setReconstruction(result.reconstruction);
    setHiddenActivations(result.hidden);

    // Отрисовка на канвасах
    if (originalCanvasRef.current) {
      drawImage(originalCanvasRef.current, sample, 28, 28);
    }
    if (reconstructionCanvasRef.current) {
      drawImage(reconstructionCanvasRef.current, result.reconstruction, 28, 28);
    }
    if (hiddenCanvasRef.current) {
      drawImage(hiddenCanvasRef.current, result.hidden, 8, 8);
    }
  }, [rbm, data, sampleIndex]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSampleIndex(parseInt(e.target.value));
  };

  const errorMetrics = currentSample && reconstruction 
    ? calculateErrorMetrics(currentSample, reconstruction)
    : null;

  const mseCategory = errorMetrics 
    ? getMSECategory(parseFloat(errorMetrics.mse))
    : null;

  return (
    <div className="visualization active">
      <div className="slider-container">
        <div className="slider-label">
          Индекс образца: <span>{sampleIndex}</span>
        </div>
        <input
          type="range"
          min="0"
          max={data.length - 1}
          value={sampleIndex}
          onChange={handleSliderChange}
        />
      </div>

      <div className="viz-panels">
        <div className="viz-panel">
          <div className="viz-title">Оригинал</div>
          <canvas
            ref={originalCanvasRef}
            width="256"
            height="256"
          />
        </div>

        <div className="viz-panel">
          <div className="viz-title">Реконструкция</div>
          <canvas
            ref={reconstructionCanvasRef}
            width="256"
            height="256"
          />
          {errorMetrics && (
            <ErrorStats 
              metrics={errorMetrics}
              category={mseCategory}
            />
          )}
        </div>

        <div className="viz-panel">
          <div className="viz-title">Активности скрытых нейронов</div>
          <canvas
            ref={hiddenCanvasRef}
            width="256"
            height="256"
          />
          <div className="hidden-neurons-explanation">
            <p>💡 <strong>Как интерпретировать:</strong></p>
            <ul>
              <li>Каждый квадрат — активность одного скрытого нейрона</li>
              <li>Светлые квадраты — нейроны с высокой активностью</li>
              <li>Темные квадраты — нейроны с низкой активностью</li>
              <li>Паттерн активности показывает, какие признаки "увидела" сеть в изображении</li>
            </ul>
          </div>
        </div>
      </div>

      {currentSample && hiddenActivations && (
        <NeuronAnalysis 
          rbm={rbm}
          sample={currentSample}
          hiddenActivations={hiddenActivations}
        />
      )}
    </div>
  );
}