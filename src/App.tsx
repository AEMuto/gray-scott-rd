import { useEffect, useRef, useState } from "react";
import styled from "styled-components";

// Shader source code
const computeShaderSource = `#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform vec2 u_resolution;
uniform float u_dA;
uniform float u_dB;
uniform float u_feed;
uniform float u_kill;
uniform float u_feedDiff;
uniform float u_feedVariation;

out vec4 fragColor;

void main() {
    vec2 texelSize = 1.0 / u_resolution;
    vec2 uv = gl_FragCoord.xy / u_resolution;
    
    // Get the current cell values
    vec4 cell = texture(u_texture, uv);
    float A = cell.r;
    float B = cell.g;
    
    // Calculate feed rate with variation
    float feedRate = u_feed + ((u_feedVariation - 50.0) * u_feedDiff) / 100.0;
    float killRate = u_kill;
    
    // Calculate Laplacian
    float laplaceA = 0.0;
    float laplaceB = 0.0;
    
    // 3x3 Laplace kernel
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            vec4 neighbor = texture(u_texture, uv + offset);
            
            float weight = 0.05;
            if(x == 0 || y == 0) weight = 0.2;
            if(x == 0 && y == 0) weight = -1.0;
            
            laplaceA += neighbor.r * weight;
            laplaceB += neighbor.g * weight;
        }
    }
    
    // Reaction-diffusion equations
    float nextA = A + (u_dA * laplaceA - A * B * B + feedRate * (1.0 - A));
    float nextB = B + (u_dB * laplaceB + A * B * B - (feedRate + killRate) * B);
    
    // Clamp values
    nextA = clamp(nextA, 0.0, 1.0);
    nextB = clamp(nextB, 0.0, 1.0);
    
    fragColor = vec4(nextA, nextB, 0.0, 1.0);
}
`;

const renderShaderSource = `#version 300 es
precision highp float;

in vec2 v_texCoord;
uniform sampler2D u_texture;
uniform float u_threshold;
uniform float u_sharpness;

out vec4 fragColor;

void main() {
    vec4 color = texture(u_texture, v_texCoord);
    float gs = color.r;
    float gs_map = (gs - u_threshold) / (0.9 - u_threshold);
    float sharp = pow(clamp(gs_map, 0.0, 1.0), u_sharpness);
    fragColor = vec4(vec3(sharp), 1.0);
}
`;

const vertexShaderSource = `#version 300 es
in vec4 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;

void main() {
    gl_Position = a_position;
    v_texCoord = a_texCoord;
}
`;

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

const createShader = (gl: WebGL2RenderingContext, type: number, source: string) => {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to create shader");

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compilation error: ${info}`);
  }

  return shader;
};

const createProgram = (
  gl: WebGL2RenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
) => {
  const program = gl.createProgram();
  if (!program) throw new Error("Failed to create program");

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error: ${info}`);
  }

  return program;
};

const DiffusionPatternRenderer = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const texturesRef = useRef<WebGLTexture[]>([]);
  const framebuffersRef = useRef<WebGLFramebuffer[]>([]);
  const computeProgramRef = useRef<WebGLProgram | null>(null);
  const renderProgramRef = useRef<WebGLProgram | null>(null);
  const [isRunning, setIsRunning] = useState(false);
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

  const initializeTexture = (gl: WebGL2RenderingContext, texture: WebGLTexture) => {
    const { width, height } = gl.canvas;
    const initialState = new Float32Array(width * height * 4);

    // Initialize with chemical A
    for (let i = 0; i < initialState.length; i += 4) {
      initialState[i] = 1; // A (red channel)
      initialState[i + 1] = 0; // B (green channel)
      initialState[i + 2] = 0; // unused
      initialState[i + 3] = 1; // alpha
    }

    // Add random spots of chemical B
    for (let i = 0; i < 20; i++) {
      const x = Math.floor(Math.random() * width);
      const y = Math.floor(Math.random() * height);
      const radius = 3;

      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy <= radius * radius) {
            const px = (x + dx + width) % width;
            const py = (y + dy + height) % height;
            const idx = (py * width + px) * 4;
            initialState[idx] = 0; // A
            initialState[idx + 1] = 1; // B
          }
        }
      }
    }

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, initialState);
  };

  const initWebGL = () => {
    if (!canvasRef.current) return;

    const gl = canvasRef.current.getContext("webgl2", { preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL 2 not supported");

    // Check for floating point texture support
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) {
      throw new Error('Floating point textures not supported');
    }

    glRef.current = gl;

    // Create compute program
    const computeVertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const computeFragmentShader = createShader(gl, gl.FRAGMENT_SHADER, computeShaderSource);
    computeProgramRef.current = createProgram(gl, computeVertexShader, computeFragmentShader);

    // Create render program
    const renderVertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const renderFragmentShader = createShader(gl, gl.FRAGMENT_SHADER, renderShaderSource);
    renderProgramRef.current = createProgram(gl, renderVertexShader, renderFragmentShader);

    // Create vertex buffer
    const vertices = new Float32Array([
      -1, -1,  0, 0,
       1, -1,  1, 0,
      -1,  1,  0, 1,
      -1,  1,  0, 1,
       1, -1,  1, 0,
       1,  1,  1, 1,
    ]);

    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    // Set up vertex attributes for both programs
    [computeProgramRef.current, renderProgramRef.current].forEach(program => {
      gl.useProgram(program);
      
      const positionLoc = gl.getAttribLocation(program, 'a_position');
      const texCoordLoc = gl.getAttribLocation(program, 'a_texCoord');
      
      gl.enableVertexAttribArray(positionLoc);
      gl.enableVertexAttribArray(texCoordLoc);
      
      gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 16, 0);
      gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 16, 8);
    });

    // Create textures and framebuffers
    texturesRef.current = [];
    framebuffersRef.current = [];

    for (let i = 0; i < 2; i++) {
      const texture = gl.createTexture();
      if (!texture) throw new Error("Failed to create texture");

      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        gl.canvas.width,
        gl.canvas.height,
        0,
        gl.RGBA,
        gl.FLOAT,
        null
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

      const fb = gl.createFramebuffer();
      if (!fb) throw new Error("Failed to create framebuffer");

      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`Framebuffer is not complete: ${status}`);
      }

      texturesRef.current.push(texture);
      framebuffersRef.current.push(fb);
    }

    // Initialize the first texture
    initializeTexture(gl, texturesRef.current[0]);

    // Set viewport
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
  };

  const computeStep = () => {
    const gl = glRef.current;
    if (!gl || !computeProgramRef.current) return;

    gl.useProgram(computeProgramRef.current);

    // Set uniforms
    const uniforms = {
      u_texture: 0,
      u_resolution: [gl.canvas.width, gl.canvas.height],
      u_dA: params.dA,
      u_dB: params.dB,
      u_feed: params.feed,
      u_kill: params.killMin,
      u_feedDiff: params.feedDiff,
      u_feedVariation: params.feedVariation,
    };

    Object.entries(uniforms).forEach(([name, value]) => {
      const location = gl.getUniformLocation(computeProgramRef.current!, name);
      if (location) {
        if (Array.isArray(value)) {
          gl.uniform2fv(location, value);
        } else if (name === 'u_texture') {
          gl.uniform1i(location, value);
        } else {
          gl.uniform1f(location, value);
        }
      }
    });

    // Ping-pong between textures
    for (let i = 0; i < params.iterations; i++) {
      const readIdx = i % 2;
      const writeIdx = (i + 1) % 2;

      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffersRef.current[writeIdx]);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texturesRef.current[readIdx]);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  };

  const render = () => {
    const gl = glRef.current;
    if (!gl || !renderProgramRef.current) return;

    gl.useProgram(renderProgramRef.current);

    // Set uniforms for render shader
    const texLocation = gl.getUniformLocation(renderProgramRef.current, "u_texture");
    gl.uniform1i(texLocation, 0);
    
    gl.uniform1f(gl.getUniformLocation(renderProgramRef.current, "u_threshold"), params.threshold);
    gl.uniform1f(gl.getUniformLocation(renderProgramRef.current, "u_sharpness"), params.sharpness);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texturesRef.current[(params.iterations + 1) % 2]);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  };


  const resetSimulation = () => {
    const gl = glRef.current;
    if (!gl || texturesRef.current.length === 0) return;

    // Stop the simulation
    setIsRunning(false);

    // Reset the first texture with new random initial state
    initializeTexture(gl, texturesRef.current[0]);

    // Clear the second texture
    gl.bindTexture(gl.TEXTURE_2D, texturesRef.current[1]);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      gl.canvas.width,
      gl.canvas.height,
      0,
      gl.RGBA,
      gl.FLOAT,
      null
    );

    // Render the initial state
    render();
  };

  useEffect(() => {
    if (!canvasRef.current) return;

    canvasRef.current.width = 400;
    canvasRef.current.height = 400;

    try {
      initWebGL();
    } catch (error) {
      console.error("WebGL initialization failed:", error);
      return;
    }

    let animationFrame: number;

    const update = () => {
      if (isRunning) {
        computeStep();
        render();
        animationFrame = requestAnimationFrame(update);
      }
    };

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
              min = "0.01";
              max = "1";
              step = "0.05";
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
