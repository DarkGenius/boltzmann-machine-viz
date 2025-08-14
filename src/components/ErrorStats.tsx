import type { ErrorMetrics } from '../types';

interface ErrorStatsProps {
  metrics: ErrorMetrics;
  category: 'excellent' | 'good' | 'poor' | null;
}

export function ErrorStats({ metrics, category }: ErrorStatsProps) {
  return (
    <div className="error-stats">
      <div className="stat-item">
        <span className="stat-label">
          MSE:
          <span className="help-icon">
            ?
            <div className="tooltip">
              <div className="tooltip-content">
                <div className={`tooltip-item ${category === 'excellent' ? 'active' : ''}`}>
                  <span className="tooltip-range">&lt; 0.01</span>
                  <span className="tooltip-desc">отличная реконструкция</span>
                </div>
                <div className={`tooltip-item ${category === 'good' ? 'active' : ''}`}>
                  <span className="tooltip-range">0.01-0.05</span>
                  <span className="tooltip-desc">хорошая реконструкция</span>
                </div>
                <div className={`tooltip-item ${category === 'poor' ? 'active' : ''}`}>
                  <span className="tooltip-range">&gt; 0.05</span>
                  <span className="tooltip-desc">заметные искажения</span>
                </div>
              </div>
            </div>
          </span>
        </span>
        <span className="stat-value">{metrics.mse}</span>
      </div>
      
      <div className="stat-item">
        <span className="stat-label">Погрешность:</span>
        <span className="stat-value">{metrics.errorPercent}</span>
      </div>
      
      <div className="stat-item">
        <span className="stat-label">Изменено пикселей:</span>
        <span className="stat-value">{metrics.pixelsDiff}</span>
      </div>
    </div>
  );
}