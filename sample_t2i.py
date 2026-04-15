from pathlib import Path
from loguru import logger
from adpdit.config import get_args
from adpdit.inference import End2End


def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` does not exist: {models_root_path}")

    # Load models
    gen = End2End(args, models_root_path)
    return args, gen


if __name__ == "__main__":
    args, gen = inferencer()

    if args.image_path is None:
        logger.error("No image path provided. Please specify an image for input.")
        raise ValueError("No image path provided.")

    logger.info("Generating images...")
    height, width = args.image_size

    results = gen.predict(
        args.prompt,
        height=height,
        width=width,
        seed=args.seed,
        enhanced_prompt=None,
        negative_prompt=args.negative,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        batch_size=args.batch_size,
        src_size_cond=args.size_cond,
        use_style_cond=args.use_style_cond,
        image_path=args.image_path,
        strength=args.strength  # Add strength parameter
    )

    images = results['images']

    # Ensure output directory is properly set
    save_dir = Path(args.results_dir) if args.results_dir else Path('results')
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory is created

    cfg_scale = args.cfg_scale
    infer_steps = args.infer_steps

    for idx, pil_img in enumerate(images):
        save_path = save_dir / f"output_cfg{cfg_scale}_steps{infer_steps}_idx{idx}.png"

        # Ensure no file is overwritten
        while save_path.exists():
            idx += 1
            save_path = save_dir / f"output_cfg{cfg_scale}_steps{infer_steps}_idx{idx}.png"

        pil_img.save(save_path)
        logger.info(f"Saved image to {save_path}")
