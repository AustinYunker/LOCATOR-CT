import os
# set environment threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import tiffs
import time, logging, argparse, yaml

from preprocess import Stage0Config
from threshold import Stage1Config
from morph_cleanup import Stage2Config
from union_bbox import Stage3Config
from refine_bbox import Stage3bConfig
from sample_localization import PipelineConfig, run_min_bounding_cuboid_pipeline


def main(args):

    # load in data
    recons = tiffs.load_stack(tiffs.list_tiffs_sorted(args.in_tiff_dir))
    print(f'\n[Info] Original Reconstruction Shape {recons.shape}')

    # read the YAML file
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)

    cfg = PipelineConfig(
            stage0=Stage0Config(
                factors=(params['stage0']['ds_z'], params['stage0']['ds_y'], params['stage0']['ds_x']),            # downsample stride
                crop_to_divisible=True,

                p_low=params['stage0']['p_low'],                    # robust normalization
                p_high=params['stage0']['p_high'],

                do_reconstruction=True,       # 2D morph reconstruction per slice
                recon_footprint_radius=1,     # small, fast footprint
                return_residual=True,         # produce residual = norm - bg
                do_circle_crop=params['stage0']['circle_crop']

            ),
            stage1=Stage1Config(
                method=params['stage1']['method'],          # or "gmm2"
                source="residual",          # same options as full-volume: try "norm" first
                use_roi_for_threshold=False,  # True if you pass roi_proxy
                air_is_low=True,        # typical CT: air is low intensity
                max_samples=20_000_000,
                clip_percentiles=(0.5, 99.5),
            ),
            stage2=Stage2Config(
                do_open=False,       # avoid fragmentation unless you know you need it
                do_close=True,
                radius=1, #default
                fill_holes_2d=True,
                min_area=0,          # start with 0; add later if needed
                z_smooth_win=params['stage2']['z_smooth_win'],      # optional; helps stability
                connectivity_2d=2
            ),
            stage3=Stage3Config(margin_zyx=(params['stage3']['margin'], params['stage3']['margin'], params['stage3']['margin'])),
            stage3b=Stage3bConfig(connectivity=params['stage3b']['connectivity'], min_largest_size=0),
        max_workers=args.n_cpus
    )
    
    START_TIME = time.perf_counter()
    out = run_min_bounding_cuboid_pipeline(V_full=recons, roi_proxy=None, cfg=cfg)
    start_slice = out.bbox_full_from_proxy_refined.z0
    print(start_slice)
    print(f'[Info] Cropped Reconstruction Shape {out.V_crop.shape}')
    print(f"\n[Info] Program Run Time: {time.perf_counter()-START_TIME}")

    tiffs.save_stack(args.out_tiff_dir, out.V_crop)


if __name__ == '__main__':

    #python main.py --n_cpus=12 --config=/home/beams/AYUNKER/APS/CAutoSL/config.yaml --in_tiff_dir=/home/beams/AYUNKER/APS/data/Plumb/A/scan1/denoised --out_tiff_dir=/home/beams/AYUNKER/APS/CAutoSL/data/plumb

    parser = argparse.ArgumentParser(description='Sample Localization for CT')
    parser.add_argument('--in_tiff_dir',   type=str, required=True, help='input directory of tiff images')
    parser.add_argument('--out_tiff_dir',   type=str, required=True, help='output directory of tiff images')
    parser.add_argument('--config', type=str, required=True, help='path to config yaml file')
    parser.add_argument('--n_cpus', type=int, required=True, help='number of cpus to use')
    
    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ...' % '\n'.join(unparsed))
        exit(0)

    main(args)