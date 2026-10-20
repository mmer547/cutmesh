"""Generate the stirred-tank mesh and save slice plots (slice_z.png, slice_y.png)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cutmesh.mesher import run
from cutmesh.viz import plot_slice

cfg = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "meshDict.json")
if not os.path.isfile(os.path.join(HERE, "tank.stl")):
    os.system(f"{sys.executable} {os.path.join(HERE, 'make_geometry.py')}")
R = run(cfg)

fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
plot_slice(ax[0], R, 2, 0.1001, "z = 0.10 (impeller mid-plane)")
plot_slice(ax[1], R, 2, 0.1001, "z = 0.10, close-up")
ax[1].set_xlim(-0.08, 0.08); ax[1].set_ylim(-0.08, 0.08)
fig.suptitle("black: STL walls   red: rotor interface (faceZone)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "slice_z.png"), dpi=170)

fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
plot_slice(ax[0], R, 1, 0.0001, "y = 0 (vertical section through blades)")
plot_slice(ax[1], R, 1, 0.0001, "y = 0, close-up of rotor zone")
ax[1].set_xlim(-0.09, 0.09); ax[1].set_ylim(0.04, 0.16)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "slice_y.png"), dpi=170)
print("slices written")
