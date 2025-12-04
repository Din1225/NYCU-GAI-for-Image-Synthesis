from huggingface_hub import snapshot_download
from pathlib import Path

# 可依需求調整下載位置
model_dir = Path("../cache_models/sdxl-base-1.0")
model_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_dir=str(model_dir),
    local_dir_use_symlinks=False,  # 確保完整檔案在專案內
    revision="main",
    token=None,  # 若已設環境變數或需要私有權限，在此填入或留 None
)
print("downloaded to", model_dir)


import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline

# 路徑設定（請依你的檔案位置調整）
model_dir = Path("../cache_models/sdxl-base-1.0")  # SDXL Base 權重路徑
prompts_file = Path("/home/a00164/oscar50513.ii13/mg/genai/T2I-CompBench/examples/dataset/texture.txt")  # 提示詞檔案，每行一條
category = "texture"  # color / shape / spatial 等
output_dir = Path("outputs") / category
output_dir.mkdir(parents=True, exist_ok=True)

# 載入管線
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_dir,
    torch_dtype=dtype,
    use_safetensors=True,
    variant="fp16",
    add_watermarker=False,
)
pipe = pipe.to(device)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

# 讀取提示詞；若檔案不存在則用範例
# if prompts_file.exists():
#     prompts = [ln.strip() for ln in prompts_file.read_text().splitlines() if ln.strip()]
# else:
#     prompts = ["a cozy watercolor house in the forest, daytime"]
prompts = [ln.strip() for ln in prompts_file.read_text().splitlines() if ln.strip()]

seed = 42
generator = torch.Generator(device=device).manual_seed(seed)

# 逐行生成，檔名依行號排序
for idx, prompt in enumerate(prompts):
    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    out_path = output_dir / f"{idx:04d}.png"
    image.save(out_path)
    print(f"{idx:04d} saved to {out_path}")
