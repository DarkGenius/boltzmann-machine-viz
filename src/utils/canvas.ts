export type ColorMap = 'gray' | 'bwr' | 'hot';

export function drawImage(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  width: number,
  _height: number
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  const imageData = ctx.createImageData(canvas.width, canvas.height);
  const scale = canvas.width / width;
  
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const srcX = Math.floor(x / scale);
      const srcY = Math.floor(y / scale);
      const value = data[srcY * width + srcX] * 255;
      const idx = (y * canvas.width + x) * 4;
      
      imageData.data[idx] = value;
      imageData.data[idx + 1] = value;
      imageData.data[idx + 2] = value;
      imageData.data[idx + 3] = 255;
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
}

export function drawImageColored(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  width: number,
  _height: number,
  colormap: ColorMap = 'gray'
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  const imageData = ctx.createImageData(canvas.width, canvas.height);
  const scale = canvas.width / width;
  
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const srcX = Math.floor(x / scale);
      const srcY = Math.floor(y / scale);
      const value = data[srcY * width + srcX];
      const idx = (y * canvas.width + x) * 4;
      
      if (colormap === 'bwr') {
        const normalized = Math.max(-1, Math.min(1, value));
        if (normalized < 0) {
          const intensity = Math.abs(normalized);
          imageData.data[idx] = 255 * (1 - intensity);
          imageData.data[idx + 1] = 255 * (1 - intensity);
          imageData.data[idx + 2] = 255;
        } else {
          const intensity = normalized;
          imageData.data[idx] = 255;
          imageData.data[idx + 1] = 255 * (1 - intensity);
          imageData.data[idx + 2] = 255 * (1 - intensity);
        }
      } else if (colormap === 'hot') {
        const v = Math.max(0, Math.min(1, value));
        if (v < 0.33) {
          imageData.data[idx] = v * 3 * 255;
          imageData.data[idx + 1] = 0;
          imageData.data[idx + 2] = 0;
        } else if (v < 0.66) {
          imageData.data[idx] = 255;
          imageData.data[idx + 1] = (v - 0.33) * 3 * 255;
          imageData.data[idx + 2] = 0;
        } else {
          imageData.data[idx] = 255;
          imageData.data[idx + 1] = 255;
          imageData.data[idx + 2] = (v - 0.66) * 3 * 255;
        }
      } else {
        const grayValue = value * 255;
        imageData.data[idx] = grayValue;
        imageData.data[idx + 1] = grayValue;
        imageData.data[idx + 2] = grayValue;
      }
      imageData.data[idx + 3] = 255;
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
}