import { useState, useEffect } from 'react';
import type { TrainingProgress, DataSource, TrainingMethod } from '../types';
import { ProgressBar } from './ProgressBar';
import { MethodAnalysis } from './MethodAnalysis';
import { DividerLine } from './DividerLine';
import { cdAnalysis, equilibriumAnalysis } from '../data/methodAnalysis';
import { DEFAULT_SAMPLE_DIGIT, DEFAULT_EPOCHS } from '../constants';

interface TrainingControlsProps {
  onTrain: () => void;
  onLoadSaved: () => void;
  onSaveWeightsToggle?: (enabled: boolean) => void;
  onDataSourceToggle: (dataSource: DataSource) => void;
  onTrainingMethodChange: (method: TrainingMethod) => void;
  onSelectedDigitChange?: (digit: number) => void;
  onEpochsChange?: (epochs: number) => void;
  isTraining: boolean;
  trainingProgress: TrainingProgress | null;
}

export function TrainingControls({
  onTrain,
  onLoadSaved,
  onSaveWeightsToggle,
  onDataSourceToggle,
  onTrainingMethodChange,
  onSelectedDigitChange,
  onEpochsChange,
  isTraining,
  trainingProgress
}: TrainingControlsProps) {
  const [saveWeights, setSaveWeights] = useState(true);
  const [useRealMNIST, setUseRealMNIST] = useState(false);
  const [trainingMethod, setTrainingMethod] = useState<TrainingMethod>('contrastive-divergence');
  const [hasSavedWeights, setHasSavedWeights] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [showCDAnalysis, setShowCDAnalysis] = useState(false);
  const [selectedDigit, setSelectedDigit] = useState(DEFAULT_SAMPLE_DIGIT);
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);

  useEffect(() => {
    const saved = localStorage.getItem('rbm_weights');
    setHasSavedWeights(!!saved);
  }, []);

  useEffect(() => {
    localStorage.setItem('rbm_save_weights', saveWeights.toString());
    onSaveWeightsToggle?.(saveWeights);
  }, [saveWeights, onSaveWeightsToggle]);

  useEffect(() => {
    // Оба метода могут использовать как MNIST, так и сгенерированные данные
    onDataSourceToggle(useRealMNIST ? 'mnist' : 'generated');
  }, [useRealMNIST, onDataSourceToggle]);

  useEffect(() => {
    onTrainingMethodChange(trainingMethod);
  }, [trainingMethod, onTrainingMethodChange]);

  useEffect(() => {
    if (onSelectedDigitChange) {
      onSelectedDigitChange(selectedDigit);
    }
  }, [selectedDigit, onSelectedDigitChange]);

  useEffect(() => {
    if (onEpochsChange) {
      onEpochsChange(epochs);
    }
  }, [epochs, onEpochsChange]);

  const handleSaveWeightsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSaveWeights(e.target.checked);
  };

  const handleDataSourceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUseRealMNIST(e.target.checked);
  };

  const handleDigitChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedDigit(Number(e.target.value));
  };

  const handleEpochsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEpochs(Number(e.target.value));
  };

  const handleTrainingMethodChange = (method: TrainingMethod) => {
    setTrainingMethod(method);
  };

  const handleDeleteWeights = () => {
    localStorage.removeItem('rbm_weights');
    setHasSavedWeights(false);
  };

  return (
    <div className="controls">
      <div className="control-section">
        <div className="section-header">⚙️ Метод обучения</div>
        <div className="method-cards">
          <div 
            className={`method-card ${trainingMethod === 'equilibrium' ? 'selected' : ''}`}
            onClick={() => !isTraining && handleTrainingMethodChange('equilibrium')}
          >
            <div className="method-card-header">
              <h3>Сэмплирование из равновесия (equilibrium sampling)</h3>
              <div className="method-year">1985</div>
            </div>
            <p className="method-description">
              Оригинальный метод из работы Хинтона и Сейновски, использующий сэмплирование из равновесия
            </p>
            <MethodAnalysis
              isVisible={showAnalysis}
              onToggle={() => setShowAnalysis(!showAnalysis)}
              isDisabled={isTraining}
              pros={equilibriumAnalysis.pros}
              cons={equilibriumAnalysis.cons}
            />
            <DividerLine />
            <div className="card-settings">
              <div className="digit-selector">
                <label className="digit-label">
                  Цифра для обучения:
                  <select 
                    value={selectedDigit} 
                    onChange={handleDigitChange}
                    disabled={isTraining}
                    className="digit-select"
                  >
                    {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(digit => (
                      <option key={digit} value={digit}>
                        {digit}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
          </div>

          <div 
            className={`method-card ${trainingMethod === 'contrastive-divergence' ? 'selected' : ''}`}
            onClick={() => !isTraining && handleTrainingMethodChange('contrastive-divergence')}
          >
            <div className="method-card-header">
              <h3>Контрастивная дивергенция (CD)</h3>
              <div className="method-year">2002</div>
            </div>
            <p className="method-description">
              Быстрый современный метод обучения RBM, предложенный Джеффри Хинтоном
            </p>
            <MethodAnalysis
              isVisible={showCDAnalysis}
              onToggle={() => setShowCDAnalysis(!showCDAnalysis)}
              isDisabled={isTraining}
              pros={cdAnalysis.pros}
              cons={cdAnalysis.cons}
            />
            <DividerLine />
            <div className="card-settings">
              {/* Настройки перенесены в общий блок */}
            </div>
          </div>
        </div>
      </div>

      <div className="control-section">
        <div className="section-header">💾 Настройки</div>
        <div className="settings-group">
          <label className="checkbox-container">
            <input
              type="checkbox"
              checked={saveWeights}
              onChange={handleSaveWeightsChange}
            />
            <span className="checkbox-label">Сохранить веса в Local Storage</span>
          </label>
          <label className="checkbox-container">
            <input
              type="checkbox"
              checked={useRealMNIST}
              onChange={handleDataSourceChange}
            />
            <span className="checkbox-label">Использовать реальные данные <span className="term-highlight" data-tooltip="Modified National Institute of Standards and Technology - стандартный набор данных для тестирования алгоритмов машинного обучения, содержащий рукописные цифры">MNIST</span></span>
          </label>
          <div className="setting-item">
            <label className="setting-label">
              Количество эпох обучения: <span className="epochs-value">{epochs}</span>
            </label>
            <input
              type="range"
              min="1"
              max="30"
              value={epochs}
              onChange={handleEpochsChange}
              disabled={isTraining}
              className="epochs-slider"
            />
            <div className="slider-labels">
              <span>1</span>
              <span>30</span>
            </div>
          </div>
        </div>
      </div>

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

      {trainingProgress && (
        <ProgressBar 
          progress={trainingProgress.progress}
          status={trainingProgress.status}
        />
      )}
    </div>
  );
}