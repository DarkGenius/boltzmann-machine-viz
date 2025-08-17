import { useEffect, useRef, useState } from 'react';
import { BernoulliRBM } from '../ml/BernoulliRBM';
import { drawImage } from '../utils/canvas';

interface FiltersVisualizationProps {
  rbm: BernoulliRBM;
}

export function FiltersVisualization({ rbm }: FiltersVisualizationProps) {
  const [previewFilter, setPreviewFilter] = useState<number | null>(null);
  const [previewPosition, setPreviewPosition] = useState<{ x: number; y: number } | null>(null);
  const [previewFilterData, setPreviewFilterData] = useState<Float32Array | null>(null);
  
  const gridRef = useRef<HTMLDivElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    drawFilters();
  }, [rbm]);

  // Отрисовка превью когда canvas готов
  useEffect(() => {
    if (previewFilterData && previewCanvasRef.current) {
      drawImage(previewCanvasRef.current, previewFilterData, 28, 28);
    }
  }, [previewFilterData, previewFilter]);

  const drawFilters = () => {
    if (!gridRef.current || !rbm) return;

    gridRef.current.innerHTML = '';
    const weights = rbm.getWeights();

    for (let i = 0; i < 64; i++) {
      const filterItem = document.createElement('div');
      filterItem.className = 'filter-item';
      
      const canvas = document.createElement('canvas');
      canvas.className = 'filter-canvas';
      canvas.width = 56;
      canvas.height = 56;

      // Нормализация весов для визуализации
      const filter = new Float32Array(784);
      let min = Infinity;
      let max = -Infinity;
      
      for (let j = 0; j < 784; j++) {
        filter[j] = weights[i][j];
        min = Math.min(min, filter[j]);
        max = Math.max(max, filter[j]);
      }
      
      // Нормализация в диапазон [0, 1]
      const range = max - min;
      if (range > 1e-8) { // Избегаем деления на ноль
        for (let j = 0; j < 784; j++) {
          filter[j] = (filter[j] - min) / range;
        }
      } else {
        // Если все веса одинаковые, устанавливаем среднее значение
        console.log(`Filter ${i}: All weights are similar (range: ${range}), min: ${min}, max: ${max}`);
        for (let j = 0; j < 784; j++) {
          filter[j] = 0.5;
        }
      }
      
      drawImage(canvas, filter, 28, 28);
      filterItem.appendChild(canvas);

      // Добавляем обработчики событий
      filterItem.addEventListener('mouseenter', (e) => {
        setPreviewFilter(i);
        setPreviewFilterData(filter);
        updatePreviewPosition(e as MouseEvent);
      });

      filterItem.addEventListener('mousemove', (e) => {
        updatePreviewPosition(e as MouseEvent);
      });

      filterItem.addEventListener('mouseleave', () => {
        setPreviewFilter(null);
        setPreviewPosition(null);
        setPreviewFilterData(null);
      });

      gridRef.current.appendChild(filterItem);
    }
  };


  const updatePreviewPosition = (e: MouseEvent) => {
    const x = e.clientX;
    const y = e.clientY;
    const previewWidth = 286; // 256px canvas + padding
    const previewHeight = 280; // canvas + title + padding
    
    // Определяем оптимальную позицию
    let left = x + 20;
    let top = y - previewHeight / 2;
    
    // Проверяем границы экрана
    if (left + previewWidth > window.innerWidth) {
      left = x - previewWidth - 20;
    }
    if (top < 10) {
      top = 10;
    }
    if (top + previewHeight > window.innerHeight - 10) {
      top = window.innerHeight - previewHeight - 10;
    }
    
    setPreviewPosition({ x: left, y: top });
  };

  return (
    <div className="visualization active filters-visualization">
      <div className="filters-container">
        <div className="filters-explanation">
          <h3>📊 Что такое фильтры?</h3>
          <p>
            Фильтры — это визуализация весов связей между входным слоем и каждым скрытым нейроном. 
            Каждый квадрат показывает, какие паттерны "выучил" конкретный нейрон.
          </p>
          <p><strong>Как интерпретировать:</strong></p>
          <ul>
            <li>Светлые области — признаки, на которые нейрон реагирует положительно</li>
            <li>Темные области — признаки, которые нейрон "игнорирует"</li>
            <li>Паттерны могут напоминать части цифр: штрихи, закругления, углы</li>
          </ul>
          <p>💡 <em>Наведите курсор на фильтр для увеличенного просмотра</em></p>
        </div>
        
        <div className="filters-grid" ref={gridRef} />
      </div>
      
      {previewFilter !== null && previewPosition && (
        <div 
          className="filter-preview active"
          style={{ 
            left: previewPosition.x,
            top: previewPosition.y 
          }}
        >
          <canvas 
            ref={previewCanvasRef}
            className="filter-preview-canvas" 
            width="256"
            height="256"
          />
          <div className="filter-preview-title">
            Фильтр #{previewFilter + 1}
          </div>
        </div>
      )}
    </div>
  );
}