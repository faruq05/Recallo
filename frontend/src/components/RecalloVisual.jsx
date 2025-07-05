import React, { useRef, useEffect } from 'react';

const RecalloVisual = () => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.style.position = 'absolute';
    canvas.style.top = '50%';
    canvas.style.left = '50%';
    canvas.style.transform = 'translate(-50%, -50%)';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '10';

    const size = 300;
    canvas.width = size;
    canvas.height = size;

    const centerX = size / 2;
    const centerY = size / 2;

    const lineCount = 40;
    const pointCount = 100;

    let tick = 0;

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < lineCount; i++) {
        const phase = (i / lineCount) * Math.PI * 2;
        ctx.beginPath();

        for (let j = 0; j <= pointCount; j++) {
          const angle = (j / pointCount) * Math.PI * 2;
          const wobble = Math.sin(angle * 3 + tick * 0.02 + phase) * 15;
          const radius = 80 + wobble;

          const x = centerX + Math.cos(angle) * radius;
          const y = centerY + Math.sin(angle) * radius;

          if (j === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.closePath();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      tick++;
    };

    draw();

    return () => cancelAnimationFrame(animationRef.current);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        backgroundColor: 'transparent',
      }}
    />
  );
};

export default RecalloVisual;
