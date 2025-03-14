# Synthetic Data Generation

We support generating assets from the objaverse datasets for simulation.

First, setup the objaverse renderer, which uses blender. I tested it with blender-3.2.2, but other versions may work as well.

```bash
pip install objaverse

cd submodules/objaverse_renderer
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar xvf blender-3.2.2-linux-x64.tar.xz
```

Generate trainable format from objaverse. Besides these UIDs examples, you can find
more UIDS in the [objaverse website](https://objaverse.allenai.org/explore/).

```bash
python objaverse_to_nerf.py --uids ecb91f433f144a7798724890f0528b23 570b82c4391c49ddb1e471e6e55de9f4 ysqP6VH2x99bODrQRbcYdQjSLjV
```

Subsequent steps (feature generation, training, etc.) are the same as the [README](../README.md).
