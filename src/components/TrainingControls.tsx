import { useState, useEffect } from 'react';
import type { TrainingProgress, DataSource, TrainingMethod } from '../types';
import { ProgressBar } from './ProgressBar';

interface TrainingControlsProps {
  onTrain: () => void;
  onLoadSaved: () => void;
  onSaveWeightsToggle: (enabled: boolean) => void;
  onDataSourceToggle: (dataSource: DataSource) => void;
  onTrainingMethodChange: (method: TrainingMethod) => void;
  isTraining: boolean;
  trainingProgress: TrainingProgress | null;
}

export function TrainingControls({
  onTrain,
  onLoadSaved,
  onSaveWeightsToggle,
  onDataSourceToggle,
  onTrainingMethodChange,
  isTraining,
  trainingProgress
}: TrainingControlsProps) {
  const [saveWeights, setSaveWeights] = useState(true);
  const [useRealMNIST, setUseRealMNIST] = useState(false);
  const [trainingMethod, setTrainingMethod] = useState<TrainingMethod>('contrastive-divergence');
  const [hasSavedWeights, setHasSavedWeights] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem('rbm_weights');
    setHasSavedWeights(!!saved);
  }, []);

  useEffect(() => {
    localStorage.setItem('rbm_save_weights', saveWeights.toString());
    onSaveWeightsToggle(saveWeights);
  }, [saveWeights, onSaveWeightsToggle]);

  useEffect(() => {
    onDataSourceToggle(useRealMNIST ? 'mnist' : 'generated');
  }, [useRealMNIST, onDataSourceToggle]);

  useEffect(() => {
    onTrainingMethodChange(trainingMethod);
  }, [trainingMethod, onTrainingMethodChange]);

  const handleSaveWeightsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSaveWeights(e.target.checked);
  };

  const handleDataSourceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUseRealMNIST(e.target.checked);
  };

  const handleTrainingMethodChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setTrainingMethod(e.target.value as TrainingMethod);
  };

  const handleDeleteWeights = () => {
    localStorage.removeItem('rbm_weights');
    setHasSavedWeights(false);
  };

  return (
    <div className="controls">
      <div className="control-section">
        <div className="section-header">🚀 Действия</div>
        <div className="action-buttons">
          <button 
            onClick={onTrain}
            disabled={isTraining}
            className="train-btn"
          >
            {isTraining ? 'Обучение...' : 'Обучить сеть'}
          </button>

          {hasSavedWeights && !isTraining && (
            <>
              <button 
                onClick={onLoadSaved}
                className="train-btn load-btn"
              >
                Загрузить сохраненные веса
              </button>
              <button 
                onClick={handleDeleteWeights}
                className="train-btn delete-btn"
              >
                🗑️ Удалить веса
              </button>
            </>
          )}
        </div>
      </div>
      
      <div className="control-section">
        <div className="section-header">⚙️ Настройки</div>
        <div className="settings-group">
          <div className="setting-item">
            <label className="setting-label">Метод обучения:</label>
            <select
              value={trainingMethod}
              onChange={handleTrainingMethodChange}
              disabled={isTraining}
              className="method-select"
            >
              <option value="contrastive-divergence">
                Контрастивная дивергенция (CD)
              </option>
              <option value="simulated-annealing">
                Имитация отжига (SA)
              </option>
            </select>
            <div className="method-description">
              {trainingMethod === 'contrastive-divergence' ? (
                <span>Быстрый современный метод (2002, Хинтон)</span>
              ) : (
                <span>Оригинальный метод из работы 1985 года (Хинтон и Сейновски)</span>
              )}
            </div>
          </div>

          <label className="checkbox-container">
            <input
              type="checkbox"
              checked={useRealMNIST}
              onChange={handleDataSourceChange}
              disabled={isTraining}
            />
            <span className="checkbox-label">
              Использовать реальные данные{' '}
              <span 
                className="term-highlight" 
                data-tooltip="MNIST (Modified National Institute of Standards and Technology) — стандартный датасет для машинного обучения, содержащий 70,000 изображений рукописных цифр от 0 до 9. Каждое изображение имеет размер 28×28 пикселей в оттенках серого."
              >
                MNIST
              </span>
            </span>
          </label>

          <label className="checkbox-container">
            <input
              type="checkbox"
              checked={saveWeights}
              onChange={handleSaveWeightsChange}
            />
            <span className="checkbox-label">Сохранить веса в Local Storage</span>
          </label>
        </div>
      </div>

      {trainingProgress && (
        <ProgressBar 
          progress={trainingProgress.progress}
          status={trainingProgress.status}
        />
      )}
    </div>
  );
}