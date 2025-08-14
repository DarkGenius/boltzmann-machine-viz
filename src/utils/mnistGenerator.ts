function drawLine(
  canvas: Float32Array,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  intensity: number,
  thickness: number,
  angle: number,
  offsetX: number,
  offsetY: number
): void {
  const steps = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1)) * 2;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = x1 + (x2 - x1) * t;
    const y = y1 + (y2 - y1) * t;
    
    const xRot = (x - 14) * Math.cos(angle) - (y - 14) * Math.sin(angle) + 14 + offsetX;
    const yRot = (x - 14) * Math.sin(angle) + (y - 14) * Math.cos(angle) + 14 + offsetY;
    
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        const px = Math.round(xRot + dx);
        const py = Math.round(yRot + dy);
        if (px >= 0 && px < 28 && py >= 0 && py < 28) {
          const dist = Math.sqrt(dx * dx + dy * dy);
          const value = Math.exp(-dist * dist / (thickness * thickness)) * intensity;
          canvas[py * 28 + px] = Math.min(1, canvas[py * 28 + px] + value);
        }
      }
    }
  }
}

function drawEllipse(
  canvas: Float32Array,
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  intensity: number,
  filled: boolean,
  thickness: number,
  angle: number,
  offsetX: number,
  offsetY: number
): void {
  for (let a = 0; a < Math.PI * 2; a += 0.05) {
    const x = cx + rx * Math.cos(a);
    const y = cy + ry * Math.sin(a);
    const nextX = cx + rx * Math.cos(a + 0.05);
    const nextY = cy + ry * Math.sin(a + 0.05);
    drawLine(canvas, x, y, nextX, nextY, intensity, thickness, angle, offsetX, offsetY);
  }
  
  if (filled) {
    for (let y = cy - ry; y <= cy + ry; y += 0.5) {
      for (let x = cx - rx; x <= cx + rx; x += 0.5) {
        if ((x - cx) * (x - cx) / (rx * rx) + (y - cy) * (y - cy) / (ry * ry) <= 1) {
          const px = Math.round(x);
          const py = Math.round(y);
          if (px >= 0 && px < 28 && py >= 0 && py < 28) {
            canvas[py * 28 + px] = Math.min(1, canvas[py * 28 + px] + intensity * 0.3);
          }
        }
      }
    }
  }
}

export function generateDigit(digit: number, variation = 0): Float32Array {
  const canvas = new Float32Array(784);
  
  const angle = (variation * 0.1 - 0.5) * 0.3;
  const thickness = 1.5 + variation * 0.2;
  const offsetX = (variation * 0.2 - 0.1) * 4;
  const offsetY = (variation * 0.2 - 0.1) * 4;
  
  const draw = (x1: number, y1: number, x2: number, y2: number, intensity = 0.9) => 
    drawLine(canvas, x1, y1, x2, y2, intensity, thickness, angle, offsetX, offsetY);
  
  const ellipse = (cx: number, cy: number, rx: number, ry: number, intensity = 0.9, filled = false) =>
    drawEllipse(canvas, cx, cy, rx, ry, intensity, filled, thickness, angle, offsetX, offsetY);
  
  switch(digit) {
    case 0:
      ellipse(14, 14, 5, 7, 0.9);
      break;
    case 1:
      draw(14, 6, 14, 22, 0.9);
      draw(14, 6, 11, 9, 0.7);
      break;
    case 2:
      draw(9, 10, 13, 6, 0.8);
      draw(13, 6, 17, 8, 0.8);
      draw(17, 8, 14, 14, 0.8);
      draw(14, 14, 9, 22, 0.8);
      draw(9, 22, 19, 22, 0.8);
      break;
    case 3:
      draw(9, 8, 17, 8, 0.8);
      draw(17, 8, 13, 14, 0.8);
      draw(13, 14, 17, 14, 0.7);
      draw(17, 14, 17, 18, 0.8);
      draw(17, 18, 13, 22, 0.8);
      draw(13, 22, 9, 20, 0.8);
      break;
    case 4:
      draw(16, 6, 10, 16, 0.8);
      draw(10, 16, 20, 16, 0.8);
      draw(16, 6, 16, 22, 0.9);
      break;
    case 5:
      draw(18, 8, 10, 8, 0.8);
      draw(10, 8, 10, 14, 0.8);
      draw(10, 14, 16, 12, 0.8);
      draw(16, 12, 18, 14, 0.8);
      draw(18, 14, 18, 18, 0.8);
      draw(18, 18, 14, 22, 0.8);
      draw(14, 22, 9, 21, 0.8);
      break;
    case 6:
      draw(16, 8, 12, 8, 0.8);
      draw(12, 8, 10, 14, 0.8);
      draw(10, 14, 10, 18, 0.8);
      draw(10, 18, 14, 22, 0.8);
      draw(14, 22, 18, 18, 0.8);
      draw(18, 18, 18, 15, 0.8);
      draw(18, 15, 14, 14, 0.8);
      draw(14, 14, 10, 15, 0.8);
      break;
    case 7:
      draw(9, 8, 19, 8, 0.9);
      draw(19, 8, 14, 22, 0.9);
      draw(12, 14, 16, 14, 0.5);
      break;
    case 8:
      ellipse(14, 10, 4, 4, 0.8);
      ellipse(14, 18, 4.5, 4, 0.8);
      break;
    case 9:
      ellipse(14, 11, 4, 4, 0.8);
      draw(18, 11, 18, 16, 0.8);
      draw(18, 16, 16, 20, 0.8);
      draw(16, 20, 12, 21, 0.8);
      break;
  }
  
  const blurred = new Float32Array(784);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      let sum = 0;
      let count = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) {
            const weight = (dx === 0 && dy === 0) ? 4 : 1;
            sum += canvas[ny * 28 + nx] * weight;
            count += weight;
          }
        }
      }
      blurred[y * 28 + x] = sum / count;
      blurred[y * 28 + x] += (Math.random() - 0.5) * 0.05;
      blurred[y * 28 + x] = Math.max(0, Math.min(1, blurred[y * 28 + x]));
    }
  }
  
  return blurred;
}

export async function loadMNIST(): Promise<Float32Array[]> {
  const nSamples = 2000;
  const data: Float32Array[] = [];
  
  for (let i = 0; i < nSamples; i++) {
    const digit = Math.floor(i / 200);
    const variation = (i % 200) / 200;
    const randomVariation = variation + Math.random() * 0.1;
    const sample = generateDigit(digit, randomVariation);
    data.push(sample);
  }
  
  for (let i = data.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [data[i], data[j]] = [data[j], data[i]];
  }
  
  // Отладочная информация
  if (data.length > 0) {
    const firstSample = data[0];
    const nonZeroPixels = Array.from(firstSample).filter(p => p > 0).length;
    const maxValue = Math.max(...Array.from(firstSample));
    const minValue = Math.min(...Array.from(firstSample));
    
    console.log(`✅ Сгенерировано ${data.length} синтетических образцов MNIST`);
    console.log(`📊 Первый образец: ${nonZeroPixels}/784 ненулевых пикселей, диапазон [${minValue.toFixed(3)}, ${maxValue.toFixed(3)}]`);
  }
  
  return data;
}