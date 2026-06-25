# Kubric-5D

Synthetic multi-view dynamic-scene dataset generation code, developed as part of the **AnyView** project.

[AnyView project page](https://tri-ml.github.io/AnyView/) | [AnyView paper](https://tri-ml.github.io/AnyView/AnyView.pdf) | License: [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

![Kubric-5D: a scene rendered along a spiral camera trajectory](media/spiral_k5d.gif)

This repository contains the code to **generate the Kubric-5D dataset**: procedurally generated, physically simulated, multi-object scenes rendered into synchronized multi-view RGB and depth videos with full per-frame camera intrinsics and extrinsics. Kubric-5D is one of the datasets used to train and evaluate AnyView (see Table 1 of the paper).

> Note: this repository contains the dataset **generation** code only. The AnyView model (training and inference) will be released separately at [TRI-ML/AnyView-DVS](https://github.com/TRI-ML/AnyView-DVS). The generated Kubric-5D dataset itself will be linked here (see the [Dataset](#dataset) section below).

## What is Kubric-5D

Per the AnyView paper, Kubric-5D is "an enhanced variant of Kubric-4D offering significantly greater camera trajectory diversity, incorporating advanced cinematography techniques like the dolly zoom." Each scene contains multiple interacting objects with complex dynamics, rendered as synchronized multi-view videos.

Compared to Kubric-4D, the main additions are:
- A wide variety of camera trajectories (orbit, radial dolly, pan, dolly zoom, sine and lissajous, etc.) with randomized phase and velocity.
- Focal-length variation, fixed or animated per clip, enabling dolly-zoom style effects.
- Color-managed rendering via Blender's Filmic OpenColorIO config for more photographic tonemapping.
- Full per-frame camera intrinsics and extrinsics exported alongside each view.

> Note: see the [Dataset](#dataset) section for concrete statistics.

## Repository layout

| Path | Description |
| --- | --- |
| `data-gen/generate.sh` | Entry point: sets env vars and runs the renderer. |
| `data-gen/export_kub_mv.py` | Multi-view Kubric scene generator (Blender plus PyBullet). |
| `data-gen/kubric_sim.py`, `kubric_constants.py`, `data_utils.py` | Renderer wrapper and helpers. |
| `data-gen/kubric_custom/` | Vendored Google Research Kubric (Apache-2.0), with a depth race-condition bugfix. |
| `data-gen/colormanagement/` | Blender Filmic OpenColorIO config plus LUTs used for tonemapping. |
| `data-gen/gpt_mass_v4.txt` | Per-asset mass estimates for physics realism. |

## Setup

Tested with Python 3.10. Follow [these instructions](https://openexr.com/en/latest/install.html) to install the OpenEXR system library, then:
```
conda create -n gcd python=3.10
conda activate gcd
pip install bpy==3.4.0 --extra-index-url https://download.blender.org/pypi/
pip install pybullet
pip install OpenEXR
cd data-gen/kubric_custom/
pip install -e . -c constraints.txt
```
Dependency version fixes are pinned in `data-gen/kubric_custom/constraints.txt` (applied via `-c` above). `data-gen/kubric_custom` is largely [this commit](https://github.com/google-research/kubric/commit/e140e24e078d5e641c4ac10bf25743059bd059ce) of Google Research Kubric, with a minor bugfix to avoid race conditions when handling depth maps.

> Note: `requirements*.txt` still carry some dependencies from the original GCD codebase and have not yet been pruned to the generation-only set.

## Usage

`data-gen/generate.sh` sets the required environment variables and renders the raw multi-view scenes:
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OCIO=$HOME/gcd/data-gen/colormanagement/config.ocio

python export_kub_mv.py --mass_est_fp=gpt_mass_v4.txt \
--root_dp=/path/to/kubricgen/any4d/ \
--num_scenes=10000 --num_workers=16 --restart_count=10001 \
--start_idx=$START_IDX --end_idx=$END_IDX --seed=$SEED \
--num_views=16 --frame_width=576 --frame_height=384 \
--num_frames=60 --frame_rate=24 --save_depth=1 --save_coords=1 \
--render_samples_per_pixel=32 --focal_length=32 \
--fixed_alter_poses=1 --few_views=4
```
Run it from within `data-gen/` (the script invokes `export_kub_mv.py` by relative path): `cd data-gen/ && bash generate.sh [START_IDX] [END_IDX] [SEED]` (defaults 0, 99999, 9000). Edit `--root_dp` to your output directory; it is the folder that will contain `scn00000`, `scn00001`, and so on. Each scene renders 16 synchronized views (4 at 45 degrees elevation, 12 at 5 degrees), 60 frames at 24 fps, with per-view RGB, depth, segmentation, normals, optical flow, and 3D world coordinates. `OCIO` points Blender at the Filmic config so renders are tonemapped consistently.

This stage is CPU- and memory-heavy. The original GCD pipeline cleared `/tmp/` between batches to avoid disk exhaustion. See `export_kub_mv.py` for the full set of parameters and trajectory logic.

The released dataset is distributed in the **AnyData unified format** (see [Dataset](#dataset)); a converter from the raw renders to the unified format will be added to this repository.

## Dataset

The released Kubric-5D dataset is distributed in the **AnyData unified format** and consists of **10,000 scenes** (`scn00000` ... `scn09999`), each rendered from **16 synchronized camera viewpoints** (4 at high elevation, 12 at low) at **576 x 384** resolution, with **60 frames** per clip at **24 FPS**. Scenes are generated i.i.d.; total size is approximately **4.8 TB**.

Each scene provides:
- `rgb/camNN.mp4` -- color video for each of the 16 views.
- `depth/camNN.zarr` -- dense per-view depth.
- `lowdim/camNN.npz` -- per-frame camera `intrinsics` (`60 x 3 x 3`, pinhole) and `extrinsics` (`60 x 4 x 4`, cam2world).
- `metadata.json` -- scene-level metadata (modalities, camera list, resolution, framerate).

Download links will be published here (coming soon).

## Licensing

- **Kubric-5D generation code** (the files authored for this project: the contents of `data-gen/`, excluding `data-gen/kubric_custom/`): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See [`LICENSE`](LICENSE).
- **`data-gen/kubric_custom/`**: Google Research Kubric, [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (see `data-gen/kubric_custom/LICENSE`).
- **`data-gen/colormanagement/`**: Blender Filmic OpenColorIO config by Troy Sobotka (see `data-gen/colormanagement/ocio-license.txt`).

See [`NOTICE`](NOTICE) for third-party attributions.

## Citations

If you use this code or the Kubric-5D dataset, please cite AnyView:
```
@inproceedings{vanhoorick2026anyview,
    title={AnyView: Synthesizing Any Novel View in Dynamic Scenes},
    author={Van Hoorick, Basile and Chen, Dian and Iwase, Shun and Tokmakov, Pavel and Irshad, Muhammad Zubair and Vasiljevic, Igor and Gupta, Swati and Cheng, Fangzhou and Zakharov, Sergey and Guizilini, Vitor Campagnolo},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2026}
}
```

This work builds on Generative Camera Dolly (GCD):
```
@article{vanhoorick2024gcd,
    title={Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis},
    author={Van Hoorick, Basile and Wu, Rundi and Ozguroglu, Ege and Sargent, Kyle and Liu, Ruoshi and Tokmakov, Pavel and Dave, Achal and Zheng, Changxi and Vondrick, Carl},
    journal={European Conference on Computer Vision (ECCV)},
    year={2024}
}
```

And the underlying Kubric framework:
```
@inproceedings{greff2022kubric,
    title={Kubric: a scalable dataset generator},
    author={Klaus Greff and Francois Belletti and Lucas Beyer and Carl Doersch and Yilun Du and Daniel Duckworth and David J Fleet and Dan Gnanapragasam and Florian Golemo and Charles Herrmann and others},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```

## Acknowledgments

- [Kubric](https://github.com/google-research/kubric) (Google Research) for the scene generation framework.
- [Filmic Blender](https://github.com/sobotka/filmic-blender) (Troy Sobotka) for the color management configuration.
- [TCOW](https://tcow.cs.columbia.edu/) for the multi-view Kubric scene design that Kubric-4D and Kubric-5D build on.
- [Generative Camera Dolly (GCD)](https://github.com/basilevh/gcd) (Van Hoorick et al., ECCV 2024), whose Kubric-4D generation pipeline this codebase builds on.
