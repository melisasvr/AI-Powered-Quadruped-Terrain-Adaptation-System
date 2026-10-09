"""MuJoCo quadruped + terrain + Gymnasium env, all in one file (no external assets)."""
import numpy as np, mujoco, gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import gaussian_filter

KINDS = ["flat", "slope", "stairs", "rubble", "slippery"]
NROW, NCOL, RX, RY, HMAX = 40, 160, 8.0, 2.0, 0.5       # heightfield: 16 m x 4 m, 10 cm cells
XS, YS = np.linspace(-RX, RX, NCOL), np.linspace(-RY, RY, NROW)
FLAT_END, START_X, GOAL_DIST = -1.5, -2.5, 3.5
Q0 = np.array([0.0, 0.6, -1.2] * 4)                      # stand pose; leg order FR, FL, RR, RL
LEG_PHASE = np.array([0, np.pi, np.pi, 0])               # trot
F0, A0, LIFT = 3.0, 0.3, 1.5                             # trot frequency (Hz), stride amp (rad), lift factor
STRIDE_SIGN = 1.0                                        # flip to -1 if the robot walks backwards
OBS_DIM, ACT_DIM = 41, 14


def _leg(n, x, y, s):
    return f"""
    <body name="{n}_hip" pos="{x} {y} 0">
      <joint name="{n}_hip" axis="1 0 0" range="-0.6 0.6" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0 {0.06*s} 0" size="0.02" mass="0.3"/>
      <body name="{n}_thigh" pos="0 {0.06*s} 0">
        <joint name="{n}_thigh" axis="0 1 0" range="-1.5 2.5" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.022" mass="0.5"/>
        <body name="{n}_calf" pos="0 0 -0.2">
          <joint name="{n}_calf" axis="0 1 0" range="-2.7 -0.2" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.018" mass="0.2"/>
          <geom name="{n}_foot" type="sphere" pos="0 0 -0.2" size="0.025" mass="0.05"/>
        </body>
      </body>
    </body>"""


def build_xml():
    legs = "".join(_leg(n, x, y, s) for n, x, y, s in
                   [("FR", .18, -.1, -1), ("FL", .18, .1, 1), ("RR", -.18, -.1, -1), ("RL", -.18, .1, 1)])
    acts = "".join(f'<position joint="{n}_{j}" kp="60" kv="2" forcerange="-25 25"/>'
                   for n in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf"))
    return f"""<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <default><geom contype="2" conaffinity="1" friction="1 0.01 0.001"/></default>
  <asset><hfield name="terr" nrow="{NROW}" ncol="{NCOL}" size="{RX} {RY} {HMAX} 0.1"/></asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="hfield" hfield="terr" contype="1" conaffinity="1"/>
    <body name="torso" pos="{START_X} 0 0.4"><freejoint/>
      <geom type="box" size="0.2 0.08 0.04" mass="5"/>{legs}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


def make_heights(kind, rng):
    """h[row=y, col=x] in metres."""
    s = np.clip(XS[None, :] - FLAT_END, 0, None) * np.ones((NROW, 1))
    if kind == "slope":
        h = np.minimum(s, 3.0) * np.tan(np.deg2rad(rng.uniform(3, 10)))
    elif kind == "stairs":
        h = np.minimum(np.floor(s / 0.35), 10) * rng.uniform(0.02, 0.05)
    elif kind == "rubble":
        n = gaussian_filter(rng.standard_normal((NROW, NCOL)), 1.5)
        n = (n - n.min()) / (np.ptp(n) + 1e-9) * rng.uniform(0.03, 0.07)
        h = n * np.clip(s / 0.5, 0, 1)
    else:
        h = np.zeros((NROW, NCOL))
    return np.clip(h, 0, HMAX - 1e-3)


def height_at(h, x, y):
    c = int(np.clip(round((x + RX) / (2 * RX) * (NCOL - 1)), 0, NCOL - 1))
    r = int(np.clip(round((y + RY) / (2 * RY) * (NROW - 1)), 0, NROW - 1))
    return h[r, c]


class QuadrupedEnv(gym.Env):
    def __init__(self, terrain="flat", push=False, max_steps=500):
        self.terrain_kind, self.push, self.max_steps = terrain, push, max_steps
        self.m = mujoco.MjModel.from_xml_string(build_xml())
        self.d = mujoco.MjData(self.m)
        self.sub = 4
        self.dt = self.sub * self.m.opt.timestep
        self.floor = self.m.geom("floor").id
        self.feet = [self.m.geom(f"{n}_foot").id for n in ("FR", "FL", "RR", "RL")]
        self.torso = self.m.body("torso").id
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1, 1, (ACT_DIM,), np.float32)
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.m, self.d
        self.kind = self.terrain_kind if self.terrain_kind != "mixed" else str(self.rng.choice(KINDS))
        self.h = make_heights(self.kind, self.rng)
        m.hfield_data[:] = (self.h / HMAX).ravel()
        fr = 0.25 if self.kind == "slippery" else 1.0          # MuJoCo uses max() of the pair
        m.geom_friction[self.floor, 0] = fr
        for f in self.feet:
            m.geom_friction[f, 0] = fr
        mujoco.mj_resetData(m, d)
        d.qpos[7:] = Q0
        d.ctrl[:] = Q0
        mujoco.mj_forward(m, d)
        for _ in range(100):
            mujoco.mj_step(m, d)
        self.phase, self.t, self.energy = 0.0, 0, 0.0
        self.pushed = self.recovered = False
        self.push_step = 100
        return self._obs(), {}

    def contacts(self):
        c = np.zeros(4, dtype=np.float32)
        for k in range(self.d.ncon):
            g = self.d.contact[k]
            if self.floor in (g.geom1, g.geom2):
                o = g.geom2 if g.geom1 == self.floor else g.geom1
                if o in self.feet:
                    c[self.feet.index(o)] = 1.0
        return c

    def _tilt(self):
        R = self.d.xmat[self.torso].reshape(3, 3)
        return R, np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1))   # R, roll, pitch

    def _obs(self):
        d = self.d
        R, roll, pitch = self._tilt()
        v, w = R.T @ d.qvel[:3], d.qvel[3:6]                   # free joint: world lin vel, body ang vel
        x, y, z = d.qpos[:3]
        ahead = [height_at(self.h, x + a, y) - z + 0.3 for a in (0.3, 0.6, 0.9)]
        o = np.concatenate([[roll, pitch], v, w, d.qpos[7:] - Q0, d.qvel[6:] * 0.05, self.contacts(),
                            [np.sin(self.phase), np.cos(self.phase)], ahead])
        return o.astype(np.float32)

    def snapshot(self):
        R, roll, pitch = self._tilt()
        return dict(pos=self.d.xpos[self.torso].copy(), R=R.copy(), feet=self.d.geom_xpos[self.feet].copy(),
                    contacts=self.contacts(), roll=roll, pitch=pitch)

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        self.phase += 2 * np.pi * (F0 + 0.6 * a[0]) * self.dt
        amp = A0 * (1 + 0.5 * a[1])
        tgt = Q0.copy()
        for l in range(4):
            ph = self.phase + LEG_PHASE[l]
            tgt[3 * l + 1] += -STRIDE_SIGN * amp * np.sin(ph)
            tgt[3 * l + 2] += -LIFT * amp * max(0.0, np.cos(ph))
        tgt += 0.25 * a[2:]
        self.d.ctrl[:] = tgt
        e = 0.0
        for _ in range(self.sub):
            mujoco.mj_step(self.m, self.d)
            e += np.abs(self.d.actuator_force * self.d.qvel[6:]).sum() * self.m.opt.timestep
        power = e / self.dt
        self.energy += e
        self.t += 1
        if self.push and not self.pushed and self.t == self.push_step:
            self.d.qvel[1] += self.rng.choice([-1, 1]) * 0.8
            self.pushed = True
        R, roll, pitch = self._tilt()
        x, y, z = self.d.qpos[:3]
        fell = z - height_at(self.h, x, y) < 0.15 or abs(roll) > 1.0 or abs(pitch) > 1.0 or abs(y) > RY - 0.3
        if self.pushed and not fell and self.t >= self.push_step + 50:
            self.recovered = True
        vx, vy, wz = self.d.qvel[0], self.d.qvel[1], self.d.qvel[5]
        r = 2.0 * np.clip(vx, -1, 1) + 0.3 - 0.01 * power - roll ** 2 - pitch ** 2 - 0.3 * abs(vy) - 0.2 * abs(wz)
        if fell:
            r = -5.0
        info = dict(x=x, dist=x - START_X, energy=self.energy, fell=bool(fell), power=power,
                    pushed=self.pushed, recovered=self.recovered, kind=self.kind)
        return self._obs(), float(r), bool(fell), self.t >= self.max_steps, info