import { useState } from 'react';
import type { ViewMode } from './types';
import { useRBM } from './hooks/useRBM';

import { TheorySection } from './components/TheorySection';
import { TrainingControls } from './components/TrainingControls';
import { InfoPanel } from './components/InfoPanel';
import { ModeButtons } from './components/ModeButtons';
import { SampleVisualization } from './components/SampleVisualization';
import { FiltersVisualization } from './components/FiltersVisualization';

import './styles/App.css';

function App() {
  const [currentMode, setCurrentMode] = useState<ViewMode>('sample');
  
  const {
    rbm,
    mnistData,
    isTraining,
    trainingProgress,
    trainNetwork,
    loadSavedWeights
  } = useRBM();

  const handleModeChange = (mode: ViewMode) => {
    if (!isTraining && rbm) {
      setCurrentMode(mode);
    }
  };

  const handleSaveWeightsToggle = (enabled: boolean) => {
    console.log('Save weights toggled:', enabled);
  };

  return (
    <div className="container">
      <h1>🧠 Машина Больцмана</h1>
      
      <TheorySection />
      
      <TrainingControls
        onTrain={trainNetwork}
        onLoadSaved={loadSavedWeights}
        onSaveWeightsToggle={handleSaveWeightsToggle}
        isTraining={isTraining}
        trainingProgress={trainingProgress}
      />

      {!rbm && !isTraining && <InfoPanel />}

      {rbm && !isTraining && (
        <>
          <div className="controls">
            <div className="control-group">
              <ModeButtons 
                currentMode={currentMode}
                onModeChange={handleModeChange}
                disabled={isTraining}
              />
            </div>
          </div>

          {currentMode === 'sample' && mnistData && (
            <SampleVisualization rbm={rbm} data={mnistData} />
          )}

          {currentMode === 'filters' && (
            <FiltersVisualization rbm={rbm} />
          )}
        </>
      )}
    </div>
  );
}

export default App;
