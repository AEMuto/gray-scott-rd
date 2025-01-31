import { useEffect, useRef, useState } from "react";
import styled from "styled-components";

const Container = styled.div`
  padding: 20px;
`;

const Canvas = styled.canvas`
  border: 1px solid #ccc;
  border-radius: 4px;
`;

const Controls = styled.div`
  margin-top: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 16px;
  align-items: center;
`;

const Button = styled.button`
  padding: 8px 16px;
  border-radius: 4px;
  border: none;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;

  &:hover {
    opacity: 0.9;
  }
`;

const StartButton = styled(Button)`
  background-color: #3b82f6;
  color: white;

  &:hover {
    background-color: #2563eb;
  }
`;

const ResetButton = styled(Button)`
  background-color: #6b7280;
  color: white;

  &:hover {
    background-color: #4b5563;
  }
`;

const ParameterControls = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const ParameterRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const Label = styled.label`
  width: 120px;
`;

const Range = styled.input`
  flex: 1;
`;

const Value = styled.span`
  width: 60px;
  text-align: right;
`;

const DiffusionPatternRenderer = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const currentGridRef = useRef<Float32Array | null>(null);
  const nextGridRef = useRef<Float32Array | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [params, setParams] = useState({
    dA: 1.0,
    dB: 0.5,
    feed: 0.03,
    feedDiff: 0.015,
    killMin: 0.056,
    killMax: 0.059,
    iterations: 8,
    feedVariation: 50,
    sharpness: 0.1,
    threshold: 0.5,
  });

  const initGrid = () => {
    if (!canvasRef.current) return;
    const { width, height } = canvasRef.current;

    currentGridRef.current = new Float32Array(width * height * 2);
    nextGridRef.current = new Float32Array(width * height * 2);

    // Initialize with chemical A
    for (let i = 0; i < currentGridRef.current.length; i += 2) {
      currentGridRef.current[i] = 1;
      currentGridRef.current[i + 1] = 0;
    }

    // Add random spots of chemical B
    for (let i = 0; i < 20; i++) {
      const x = Math.floor(Math.random() * width);
      const y = Math.floor(Math.random() * height);
      const radius = 3;

      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const px = (x + dx + width) % width;
          const py = (y + dy + height) % height;
          if (dx * dx + dy * dy <= radius * radius) {
            const idx = (py * width + px) * 2;
            if (currentGridRef.current) {
              currentGridRef.current[idx] = 0;
              currentGridRef.current[idx + 1] = 1;
            }
          }
        }
      }
    }
  };

  const resetSimulation = () => {
    if (!canvasRef.current) return;
    currentGridRef.current = null;
    nextGridRef.current = null;
    const ctx = canvasRef.current.getContext("2d");
    initGrid();
    if (ctx) render(ctx);
  };

  const computeStep = () => {
    if (!canvasRef.current || !currentGridRef.current || !nextGridRef.current) return;

    const { width, height } = canvasRef.current;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 2;

        const normalizedX = x / width;
        const feedRate = params.feed + ((params.feedVariation - 50) * params.feedDiff) / 100;
        const killRate = params.killMin + normalizedX * (params.killMax - params.killMin);

        let laplaceA = 0;
        let laplaceB = 0;

        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const px = (x + dx + width) % width;
            const py = (y + dy + height) % height;
            const idx = (py * width + px) * 2;

            let weight = 0.05;
            if (dx === 0 || dy === 0) weight = 0.2;
            if (dx === 0 && dy === 0) weight = -1.0;

            laplaceA += currentGridRef.current[idx] * weight;
            laplaceB += currentGridRef.current[idx + 1] * weight;
          }
        }

        const a = currentGridRef.current[i];
        const b = currentGridRef.current[i + 1];

        nextGridRef.current[i] = a + (laplaceA * params.dA - a * b * b + feedRate * (1 - a));
        nextGridRef.current[i + 1] =
          b + (laplaceB * params.dB + a * b * b - (feedRate + killRate) * b);

        nextGridRef.current[i] = Math.max(0, Math.min(1, nextGridRef.current[i]));
        nextGridRef.current[i + 1] = Math.max(0, Math.min(1, nextGridRef.current[i + 1]));
      }
    }

    // Swap buffers
    const temp = currentGridRef.current;
    currentGridRef.current = nextGridRef.current;
    nextGridRef.current = temp;
  };

  const render = (ctx: CanvasRenderingContext2D) => {
    if (!currentGridRef.current) return;

    const { width, height } = ctx.canvas;
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < currentGridRef.current.length; i += 2) {
      const idx = i * 2;
      const a = currentGridRef.current[i];
      const gs = a;
      const gs_map = (gs - params.threshold) / (0.9 - params.threshold);
      const sharpValue = Math.pow(Math.max(0, Math.min(1, gs_map)), params.sharpness);
      const final = Math.round(sharpValue * 255);

      imageData.data[idx] = final;
      imageData.data[idx + 1] = final;
      imageData.data[idx + 2] = final;
      imageData.data[idx + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
  };

  useEffect(() => {
    if (!canvasRef.current) return;

    let animationFrame: number;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = 400;
    canvas.height = 400;

    if (!currentGridRef.current) {
      initGrid();
    }

    const update = () => {
      if (isRunning && ctx) {
        for (let i = 0; i < params.iterations; i++) {
          computeStep();
        }
        render(ctx);
        animationFrame = requestAnimationFrame(update);
      }
    };

    if (ctx) render(ctx);

    if (isRunning) {
      update();
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [isRunning, params]);

  return (
    <Container>
      <Canvas ref={canvasRef} />
      <Controls>
        <ButtonGroup>
          <StartButton onClick={() => setIsRunning(!isRunning)}>
            {isRunning ? "Stop" : "Start"}
          </StartButton>
          <ResetButton onClick={resetSimulation}>Reset</ResetButton>
        </ButtonGroup>

        <ParameterControls>
          {Object.entries(params).map(([key, value]) => {
            let min = "0";
            let max = "0.1";
            let step = "0.001";

            if (key === "feedVariation") {
              min = "0";
              max = "100";
              step = "1";
            } else if (key === "iterations") {
              min = "1";
              max = "20";
              step = "1";
            } else if (key === "dA" || key === "dB") {
              max = "2";
            } else if (key === "sharpness") {
              min = "0.1";
              max = "5";
              step = "0.1";
            } else if (key === "threshold") {
              min = "0";
              max = "0.9";
              step = "0.1";
            }

            return (
              <ParameterRow key={key}>
                <Label>{key}:</Label>
                <Range
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={value}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      [key]: parseFloat(e.target.value),
                    })
                  }
                />
                <Value>
                  {["iterations", "feedVariation"].includes(key) ? value : value.toFixed(3)}
                </Value>
              </ParameterRow>
            );
          })}
        </ParameterControls>
      </Controls>
    </Container>
  );
};

export default DiffusionPatternRenderer;
