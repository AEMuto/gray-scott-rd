# WebGL Reaction-Diffusion Visualizer

A high-performance implementation of the Gray-Scott reaction-diffusion model using WebGL 2.0. This project demonstrates how complex chemical reaction patterns can be simulated efficiently on the GPU.

## Features

- Real-time visualization of reaction-diffusion patterns
- GPU-accelerated computation using WebGL 2.0
- Interactive parameter controls
- Mobile-friendly
- Support for different pattern types through parameter adjustment
- High-performance even with large grid sizes

## Technical Details

The simulation implements the Gray-Scott model, which describes the interaction between two chemicals (U and V) through coupled differential equations:

```
∂u/∂t = Du∇²u - uv² + F(1-u)
∂v/∂t = Dv∇²v + uv² - (F+k)v
```

where:
- Du, Dv are diffusion rates
- F is the feed rate
- k is the kill rate
- u, v are chemical concentrations

The implementation uses:
- WebGL 2.0 for GPU acceleration
- GLSL shaders for computation and rendering
- Ping-pong texture buffers for state management
- React for UI components
- Styled-components for styling

## Controls

- **Start/Stop**: Toggle the simulation
- **Reset**: Generate new random initial conditions
- **Parameters**:
  - dA, dB: Diffusion rates
  - feed: Feed rate
  - feedDiff: Feed rate variation
  - killMin/killMax: Kill rate range
  - iterations: Simulation steps per frame
  - feedVariation: Feed rate spatial variation
  - sharpness: Output contrast
  - threshold: Visualization threshold

## Requirements

- Browser with WebGL 2.0 support
- Deno for development
- Vite for building and serving

## Installation

```bash
# Clone the repository
git clone https://github.com/AEMuto/gray-scott-rd.git

# Install dependencies
deno install

# Start development server
deno task dev
```

## Usage

Visit the application in your browser and experiment with different parameters to create various patterns:
- Spots: High feed rate, low kill rate
- Stripes: Medium feed rate, medium kill rate
- Maze-like: Low feed rate, high kill rate

## Technical Implementation

The simulation uses two main shaders:
1. Compute shader: Implements the reaction-diffusion equations
2. Render shader: Handles visualization of the chemical concentrations

The system uses ping-pong buffers to update the state, allowing for efficient GPU computation without CPU intervention.

## Performance

The WebGL implementation provides significant performance improvements over CPU-based approaches:
- Parallel processing of all cells
- GPU-optimized floating-point calculations
- Minimal CPU-GPU data transfer
- Efficient memory usage through texture ping-ponging

## Future Improvements

Potential areas for enhancement:
- Multiple pattern presets
- Pattern save/load functionality
- Custom initial condition drawing
- Color mapping options
- WebGL fallback for older devices
- Performance metrics display

## Credits

This project builds upon the work and resources from several excellent sources:

1. Daniel Shiffman's Coding Train tutorial on Reaction Diffusion:
   - https://www.youtube.com/watch?v=BV9ny785UNc

2. Karl Sims' detailed explanation of reaction-diffusion systems:
   - https://www.karlsims.com/rd.html

3. Frankfurt University's student project on reaction-diffusion:
   - https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/

Built with React, WebGL 2.0, and styled-components.
