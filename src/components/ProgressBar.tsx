interface ProgressBarProps {
  progress: number;
  status: string;
}

export function ProgressBar({ progress, status }: ProgressBarProps) {
  return (
    <div className="progress-container active">
      <div className="status-text">{status}</div>
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}