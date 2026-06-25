export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OCIO=$HOME/gcd/data-gen/colormanagement/config.ocio

START_IDX=${1:-0}
END_IDX=${2:-99999}
SEED=${3:-9000}

python export_kub_mv.py --mass_est_fp=gpt_mass_v4.txt \
--root_dp=$HOME/kubricgen/any4d/ \
--num_scenes=10000 --num_workers=16 --restart_count=10001 --start_idx=$START_IDX --end_idx=$END_IDX \
--seed=${SEED} --num_views=16 --frame_width=576 --frame_height=384 \
--num_frames=60 --frame_rate=24 --save_depth=1 --save_coords=1 \
--render_samples_per_pixel=32 --focal_length=32 \
--fixed_alter_poses=1 --few_views=4
