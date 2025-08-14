import { useState, useEffect } from 'react';
import type { TrainingProgress } from '../types';
import { ProgressBar } from './ProgressBar';

interface TrainingControlsProps {
  onTrain: () => void;
  onLoadSaved: () => void;
  onSaveWeightsToggle: (enabled: boolean) => void;
  isTraining: boolean;
  trainingProgress: TrainingProgress | null;
}

export function TrainingControls({
  onTrain,
  onLoadSaved,
  onSaveWeightsToggle,
  isTraining,
  trainingProgress
}: TrainingControlsProps) {
  const [saveWeights, setSaveWeights] = useState(true);
  const [hasSavedWeights, setHasSavedWeights] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem('rbm_weights');
    setHasSavedWeights(!!saved);
  }, []);

  useEffect(() => {
    localStorage.setItem('rbm_save_weights', saveWeights.toString());
    onSaveWeightsToggle(saveWeights);
  }, [saveWeights, onSaveWeightsToggle]);

  const handleSaveWeightsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSaveWeights(e.target.checked);
  };

  return (
    <div className="controls">
      <div className="control-group">
        <button 
          onClick={onTrain}
          disabled={isTraining}
          className="train-btn"
        >
          {isTraining ? 'Обучение...' : 'Обучить сеть'}
        </button>

        {hasSavedWeights && !isTraining && (
          <button 
            onClick={onLoadSaved}
            className="train-btn load-btn"
          >
            Загрузить сохраненные веса
          </button>
        )}

        <label className="checkbox-container">
          <input
            type="checkbox"
            checked={saveWeights}
            onChange={handleSaveWeightsChange}
          />
          <span className="checkbox-label">Сохранить веса в Local Storage</span>
        </label>
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