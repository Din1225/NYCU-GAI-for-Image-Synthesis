## Pix2Pix 報告（CIFAR-10 Airplane/Bird/Cat）

### 訓練與資料設定
- 以 CIFAR-10 中 `airplane`、`bird`、`cat` 三個類別各取 100 張（共 300 張）影像，經 `64×64` resize 與 `[-1, 1]` normalization 作為 paired data。
- Generator 採 UNet 結構、Discriminator 為 PatchGAN，皆以 Adam (lr = 2e-4 / 3e-5, betas = 0.5, 0.999) 訓練 350 epochs，GAN loss + λ=100 的 L1 loss。
- Batch size 為 4，Training/Inference/Evaluation 均僅使用專案內的 `pix2pix.ipynb` pipeline。

### Loss Curve 與訓練觀察
- 判別器平均 loss（前 10 vs 後 10 epochs）由 `0.56 → 0.094`，顯示 D 很快收斂並維持穩定。
- 生成器 adversarial loss 由 `1.20 → 5.20`，代表在 D 變強後 G 需要付出更高對抗成本；然而 L1 loss 從 `7.57 → 1.74`，圖像重建誤差持續下降。
- Loss curve 顯示兩者皆趨於平穩，無明顯 mode collapse 或梯度爆炸跡象。

![Loss Curve](assets/loss_curve.png)

### 量化結果（100 張評估子集）
- `PSNR = 40.0898`
- `SSIM = 0.9843`
- `LPIPS = 0.0308`


高 PSNR/SSIM 與低 LPIPS 代表生成影像在像素/結構/感知三面向都與目標影像高度一致，符合 pix2pix 的 paired translation 目標。

### 定性分析（每類 4 張推論結果）
- 下圖每列對應 `airplane`、`bird`、`cat`，每列 4 張樣本，展示訓練後生成器能維持類別特徵且避免顏色崩壞。
- `airplane`：機體輪廓與背景天空過渡自然，翼尖/尾翼細節可辨。
- `bird`：羽毛色塊平滑，仍可見胸腹對比；背景雜訊被抑制。
- `cat`：臉部對稱與耳朵形狀被保留，僅在毛色細節略有模糊。

![Qualitative Samples](assets/qualitative_samples.png)

### Pix2Pix vs. Auto-encoder
- **優點**：Pix2Pix 屬於 conditional GAN，訓練目標直接對應輸入與目標影像，可在生成過程中保留空間細節，對比傳統 auto-encoder 僅透過重建 loss，Pix2Pix 靠對抗訓練與 skip connections 讓輸出更銳利、能控制輸出風格；同時可以對不同條件（如類別或語意圖）產生對應影像。
- **缺點**：需要成對的資料集，收集成本高；對抗訓練較不穩定，需要平衡 G/D 以及更多超參數調整；模型參數量與計算量高於單純 auto-encoder，推論成本也較大；若 paired data 品質不一致，容易造成 artifact 或 mode collapse，而 auto-encoder 在無監督情境下可利用更多未標記資料。

### 後續可優化方向
- 增加資料量或採 progressive resizing，以降低小尺寸資料對高頻細節的限制。
- 加入 feature matching / perceptual loss，再與目前的 LPIPS 指標對照，檢視是否能進一步降低感知差距。
