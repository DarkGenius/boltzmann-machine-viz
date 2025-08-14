import type { ViewMode } from '../types';

interface ModeButtonsProps {
  currentMode: ViewMode;
  onModeChange: (mode: ViewMode) => void;
  disabled?: boolean;
}

export function ModeButtons({ currentMode, onModeChange, disabled }: ModeButtonsProps) {
  return (
    <div className="mode-buttons">
      <button
        className={`mode-btn ${currentMode === 'sample' ? 'active' : ''}`}
        onClick={() => onModeChange('sample')}
        disabled={disabled}
      >
        Образец
      </button>
      <button
        className={`mode-btn ${currentMode === 'filters' ? 'active' : ''}`}
        onClick={() => onModeChange('filters')}
        disabled={disabled}
      >
        Фильтры
      </button>
    </div>
  );
}