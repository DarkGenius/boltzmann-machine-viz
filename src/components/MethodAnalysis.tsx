import React from 'react';

interface AnalysisItemProps {
  icon: string;
  text: string;
  tooltip: string;
}

function AnalysisItem({ icon, text, tooltip }: AnalysisItemProps) {
  return (
    <div className="analysis-item" data-tooltip={tooltip}>
      <span className="item-icon">{icon}</span>
      <span className="item-text">{text}</span>
      <span className="item-arrow">→</span>
    </div>
  );
}

interface MethodAnalysisProps {
  isVisible: boolean;
  onToggle: () => void;
  isDisabled?: boolean;
  pros: AnalysisItemProps[];
  cons: AnalysisItemProps[];
}

export function MethodAnalysis({ 
  isVisible, 
  onToggle, 
  isDisabled = false, 
  pros, 
  cons 
}: MethodAnalysisProps) {
  return (
    <>
      <button 
        className="details-btn"
        onClick={(e) => {
          e.stopPropagation();
          onToggle();
        }}
        disabled={isDisabled}
      >
        {isVisible ? 'Скрыть подробности' : 'Подробнее'}
        <span className={`btn-icon ${isVisible ? 'rotated' : ''}`}>▼</span>
      </button>
      
      <div className={`method-analysis ${isVisible ? 'active' : ''}`}>
        <div className="analysis-section">
          <div className="section-title">✅ Плюсы</div>
          {pros.map((item, index) => (
            <AnalysisItem
              key={`pro-${index}`}
              icon={item.icon}
              text={item.text}
              tooltip={item.tooltip}
            />
          ))}
        </div>
        
        <div className="analysis-section">
          <div className="section-title">❌ Минусы</div>
          {cons.map((item, index) => (
            <AnalysisItem
              key={`con-${index}`}
              icon={item.icon}
              text={item.text}
              tooltip={item.tooltip}
            />
          ))}
        </div>
      </div>
    </>
  );
}
