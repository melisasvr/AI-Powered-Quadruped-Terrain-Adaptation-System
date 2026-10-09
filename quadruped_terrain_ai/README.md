# AI-Powered Quadruped Terrain Adaptation System

A simulation and control system for a four-legged robot that learns to adapt its gait to difficult terrain: stairs, slopes, rubble and slippery ground. A small neural network, trained with evolution strategies, modulates a trot gait generator to keep the robot balanced and moving forward.

![Dashboard screenshot](docs/dashboard.png)

## Features

- **Quadruped locomotion simulation** with a 12-joint robot defined in code (no external assets to download)
- **Procedural terrain generation** with randomized parameters: flat, slope, stairs, rubble and slippery surfaces
- **AI-based gait adaptation and balance control** using a JAX neural network trained with evolution strategies
- **Disturbance testing** with a lateral push to measure recovery
- **Real-time visualization** of body posture, foot contacts and stability in an interactive Streamlit dashboard with a PyVista 3D view
- **Performance benchmark** reporting success rate, distance, energy use and recovery rate per terrain

## Tech Stack

| Purpose | Library |
|---|---|
| Physics simulation | MuJoCo |
| RL environment API | Gymnasium |
| Policy and learning | JAX |
| Dashboard | Streamlit |
| 3D visualization | PyVista |
| Numerics and plots | NumPy, SciPy, Pandas, Matplotlib |

No PyTorch and no C++ build step are required.

## Project Structure

```
quadruped_terrain_ai/
├── quadruped.py       # Robot model (MJCF), terrain generation, Gymnasium environment
├── brain.py           # JAX policy network, rollouts, benchmark
├── train.py           # Evolution-strategies training (parallel rollouts)
├── dashboard.py       # Streamlit dashboard with PyVista 3D view
├── policy.npy         # Trained policy weights (created by train.py)
├── requirements.txt   # Python dependencies
├── docs/
│   └── dashboard.png  # Dashboard screenshot
├── LICENSE
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**1. Train a policy**

```bash
python train.py --iters 150 --pop 16
```

The policy is saved to `policy.npy` every 5 iterations, so training can be stopped at any time.

| Option | Default | Description |
|---|---|---|
| `--iters` | 150 | Number of training iterations |
| `--pop` | 16 | Antithetic sample pairs per iteration |
| `--sigma` | 0.05 | Noise scale for parameter perturbations |
| `--lr` | 0.03 | Learning rate |
| `--workers` | CPU count - 1 | Parallel simulation processes |
| `--out` | `policy.npy` | Output file |

**2. Benchmark on all terrains**

```bash
python brain.py --policy policy.npy --episodes 5
```

Run without a policy file to benchmark the untrained trot baseline.

**3. Launch the dashboard**

```bash
streamlit run dashboard.py
```

Pick a terrain, press **Run episode**, and use the frame slider to step through the run. The **Benchmark** tab compares all terrains. On a headless Linux server, use `xvfb-run -a streamlit run dashboard.py` so PyVista can render.

## How It Works

- **Robot.** A 9 kg quadruped with a box torso and four 3-joint legs (hip abduction, thigh, calf), driven by position actuators. It is defined as an MJCF string in `quadruped.py`.

- **Gait generator.** A central-pattern-generator trot moves diagonal leg pairs in opposition. The neural network does not control the joints directly. Instead, it adjusts the gait.

- **Policy.** An MLP (41 inputs, two hidden layers of 64, 14 outputs) outputs:
- gait frequency modulation (1 value)
- stride amplitude modulation (1 value)
- per-joint offsets on top of the gait (12 values)

- **Observation (41 values).** Roll and pitch, body-frame linear and angular velocity, joint positions and velocities, foot contacts, gait phase, and terrain height at three points ahead of the robot.

- **Reward.** Forward velocity and a survival bonus, minus penalties for energy use, body tilt, sideways drift and yaw rate. Falling ends the episode with a large penalty.

- **Training.** Antithetic evolution strategies with rank-normalized fitness and the Adam update. Simulations run in parallel across processes, and each iteration samples random terrains.

- **Terrain.** Heightfields (16 m by 4 m, 10 cm cells) with a flat start zone. Slope angle, stair height and rubble roughness are randomized per episode. Slippery terrain lowers the friction coefficient.

**Metrics.**

| Metric | Meaning |
|---|---|
| Success | Walked at least 3.5 m without falling |
| Distance | Forward distance travelled (m) |
| Energy | Total mechanical work (J) |
| Fell | Body tilted past 1 rad, dropped too low, or left the terrain |
| Push recovered | Still upright 50 control steps after the lateral push |

## Results

Benchmark after 150 training iterations (5 episodes per terrain, with push):

| Terrain | Success rate |
|---|---|
| Slippery | 100% |
| Slope | 60% |
| Flat | 40% |
| Rubble | 20% |
| Stairs | 0% |

- Stairs are not yet solved. See the roadmap below. Results vary between runs, and longer training improves them.

## Configuration and Troubleshooting

- **Robot walks backwards:** set `STRIDE_SIGN = -1` in `quadruped.py`.
- **Gait tuning:** adjust `F0`, `A0` and `LIFT` (frequency, stride amplitude, lift factor) in `quadruped.py`.
- **Terrain difficulty:** edit the ranges in `make_heights()` in `quadruped.py`.
- **Reward and termination:** edit `QuadrupedEnv.step()` in `quadruped.py`.
- **Windows:** training uses multiprocessing with the spawn method. Keep the `if __name__ == "__main__":` guard in `train.py`.
- **3D view shows a side view with a "PyVista unavailable" title:** the error type is shown in that title. Make sure `pyvista` is installed and that your graphics drivers support off-screen rendering.

## Roadmap

- Stairs curriculum (start with low steps and increase height during training)
- Resume training from an existing `policy.npy`
- Longer training runs with larger populations
- Heading control and turning
- Domain randomization (mass, friction, motor strength)

## Contributing

Contributions are welcome.

1. Fork the repository and create a branch: `git checkout -b feature/your-feature`
2. Make your changes and keep them focused on one topic.
3. Check that `python train.py --iters 2 --pop 2` and `streamlit run dashboard.py` still run.
4. Commit with a clear message and open a pull request describing what changed and why.

Please open an issue first for larger changes, and include steps to reproduce when reporting a bug.

## License
- This project is licensed under the MIT License. 
- MIT License
```
Copyright (c) 2026 Melisa Sever

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
