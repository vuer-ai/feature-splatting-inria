# Synthetic Data Generation

We support generating assets from the objaverse datasets for simulation.

First, setup the objaverse renderer:
```bash
pip install objaverse

cd submodules/objaverse_renderer
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar -xf blender-3.2.2-linux-x64.tar.xz
```

Generate trainable format from objaverse:

```bash
python objaverse_to_nerf.py
```

Subsequent steps (feature generation, training, etc.) are the same as the [README](../README.md).
