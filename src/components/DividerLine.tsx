import React from 'react';

interface DividerLineProps {
  className?: string;
}

export function DividerLine({ className = '' }: DividerLineProps) {
  return (
    <div className={`divider-line ${className}`} />
  );
}
