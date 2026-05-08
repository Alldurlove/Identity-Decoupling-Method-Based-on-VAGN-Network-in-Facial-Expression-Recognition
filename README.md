VGAN 是基于 VAE 编码器与 GAN 的对抗框架。  
本项目通过将人脸身份特征与表情特征解耦，实现“保留表情、隐藏身份”的隐私保护目标。

## 网页端部署

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 启动 Web 服务

```bash
cd /home/ubuntu/VGAN-Project
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

浏览器打开：`http://localhost:8000`

### 3) 使用真实模型（可选）

如果你有训练好的生成器权重，启动前设置环境变量：

```bash
export CHECKPOINT_PATH=/home/ubuntu/VGAN-Project/checkpoints/netG_epoch_199.pth
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

如果没有权重，系统会自动进入 `fallback` 演示模式（仍可完整展示网页端流程与交互）。

## 功能说明

- 前端：浏览器摄像头采集 + 实时调用后端推理接口  
- 后端：FastAPI 提供 `/api/health` 与 `/api/infer`  
- 静态站点：由 FastAPI 直接托管 `web/` 页面，可直接对外部署

## 隐私攻击实验（论文实验）

新增脚本位于 `attacks/`，支持三步完成隐私攻击评估：

### 1) 生成脱敏样本集

```bash
python attacks/generate_anonymized_dataset.py \
  --dataroot /path/to/FERG_DB_256 \
  --checkpoint /home/ubuntu/VGAN-Project/checkpoints/netG_epoch_199.pth \
  --output-root attack_data/anonymized
```

输出：
- `attack_data/anonymized/images/*`：生成后的脱敏图像
- `attack_data/anonymized/metadata.csv`：实验元数据（含 split/source_id/target_id）

### 2) 源身份重识别攻击（Source-ID Attack）

```bash
python attacks/train_source_id_attacker.py \
  --metadata-csv attack_data/anonymized/metadata.csv \
  --image-column anonymized_path \
  --output-dir attack_results/source_id
```

脚本会训练攻击器并输出：
- `attack_results/source_id/best_source_id_attacker.pth`
- `attack_results/source_id/source_id_attack_report.json`
- `attack_results/source_id/source_id_train_history.json`
- `attack_results/source_id/source_id_training_curves.png`

可将 `--image-column` 改为 `source_path`，得到原图上的攻击上限基线。

### 3) 链接攻击（Linkability）

```bash
python attacks/eval_linkability.py \
  --metadata-csv attack_data/anonymized/metadata.csv \
  --attacker-ckpt attack_results/source_id/best_source_id_attacker.pth \
  --image-column anonymized_path \
  --output-json attack_results/linkability_report.json
```

输出指标：
- `AUC`：越接近 `0.5` 表示越难链接
- `EER`：越接近 `0.5` 表示越难链接
- `linkability_roc.png`：ROC 曲线图（论文可直接使用）

论文建议同时报告：
- Source-ID attack：`Top-1 Accuracy / Macro-F1`
- Linkability：`AUC / EER`
- 实用性保持：脱敏图像上的表情识别准确率（可复用同样训练范式）

## 真人表情微调（中策）

### 1) 准备真人表情数据目录

先把原始图片按表情放到一个目录（至少包含以下子目录中的若干个）：

```text
raw_real_expr/
  anger/
  disgust/
  fear/
  joy/
  neutral/
  sadness/
  surprise/
```

再执行整理脚本，生成 `train/val/test/<expression>/` 结构：

```bash
python tools/prepare_real_expr_dataset.py \
  --source-root /path/to/raw_real_expr \
  --output-root /home/ubuntu/VGAN-Project/real_data \
  --split-ratio 70,15,15
```

### 2) 微调 netG/netD（两阶段）

```bash
python finetune.py \
  --real-data-root /home/ubuntu/VGAN-Project/real_data \
  --ferg-dataroot /path/to/FERG_DB_256 \
  --real-mix-ratio 0.7 \
  --resume-netg /home/ubuntu/model/netG_epoch_199.pth \
  --resume-netd /home/ubuntu/model/netD_epoch_199.pth \
  --stage-a-epochs 5 \
  --stage-b-epochs 10 \
  --batch-size 16 \
  --save-dir /home/ubuntu/VGAN-Project/checkpoints/finetune
```

输出最佳权重：
- `checkpoints/finetune/netG_finetuned_best.pth`
- `checkpoints/finetune/netD_finetuned_best.pth`

### 3) 表情保持评估

先用新的 netG 生成脱敏数据，再训练表情攻击器：

```bash
python attacks/generate_anonymized_dataset.py \
  --dataroot /path/to/FERG_DB_256 \
  --checkpoint /home/ubuntu/VGAN-Project/checkpoints/finetune/netG_finetuned_best.pth \
  --output-root attack_data/anonymized_finetuned

python attacks/train_expression_attacker.py \
  --metadata-csv attack_data/anonymized_finetuned/metadata.csv \
  --image-column anonymized_path \
  --output-dir attack_results/expression_finetuned
```

### 4) 一键对比旧权重 vs 新权重（隐私+表情）

```bash
python tools/run_checkpoint_comparison.py \
  --dataroot /path/to/FERG_DB_256 \
  --baseline-ckpt /home/ubuntu/model/netG_epoch_199.pth \
  --finetuned-ckpt /home/ubuntu/VGAN-Project/checkpoints/finetune/netG_finetuned_best.pth \
  --output-root attack_results/checkpoint_comparison
```

对比结果汇总：
- `attack_results/checkpoint_comparison/comparison_summary.json`

### 5) 部署最佳 checkpoint 到 systemd

```bash
bash tools/deploy_best_checkpoint.sh \
  /home/ubuntu/VGAN-Project/checkpoints/finetune/netG_finetuned_best.pth
```

可选：部署后快速回归检查

```bash
python tools/verify_web_runtime.py \
  --base-url http://127.0.0.1:8000 \
  --output-json runtime_verify_report.json
```

### 6) 部署前清晰度质检（推荐）

先对比旧模型与候选模型的生成清晰度/对比度；若候选低于阈值，脚本会返回非 0 并拒绝上线建议。

```bash
python tools/predeploy_quality_gate.py \
  --baseline-ckpt /home/ubuntu/model/netG_epoch_199.pth \
  --candidate-ckpt /home/ubuntu/model/netG_finetuned_best.pth \
  --image-root /path/to/your/validation_images \
  --target-id 0 \
  --max-samples 120 \
  --min-sharpness-ratio 0.85 \
  --min-contrast-ratio 0.85 \
  --output-json quality_gate_report.json
```

当输出 `pass=false` 时，建议继续训练并调整超参数，不要直接切线上模型。

## 真人重训（最小改动流水线）

### 0) Git 版本控制（强烈建议）

```bash
git checkout -b retrain-real-minimal
```

建议每个里程碑单独提交：数据适配、训练入口、评估门禁、主观AB切换、报告打包。

### 1) 固定可复现实验配置并启动训练

默认配置文件：
- `configs/retrain_real_baseline.json`

先做 dry-run 检查命令：

```bash
python tools/run_retrain_baseline.py --dry-run
```

正式执行训练：

```bash
python tools/run_retrain_baseline.py
```

可选：短轮数快速冒烟（例如 5 轮）：

```bash
python tools/run_retrain_baseline.py --epochs-override 5
```

### 2) 门禁 + 表达指标双筛选

```bash
python tools/run_gate_and_metrics.py \
  --baseline-ckpt /home/ubuntu/model/netG_epoch_199.pth \
  --candidate-ckpt /home/ubuntu/VGAN-Project/checkpoints/real_from_scratch/netG_best.pth \
  --image-root /path/to/validation_images \
  --dataroot /path/to/FERG_DB_256 \
  --summary-json attack_results/gate_metrics_summary.json
```

输出建议关注：
- `quality_gate.pass`（必须为 `true`）
- `expression_metrics.acc` / `expression_metrics.macro_f1`

### 3) 主观AB顺序切换与回滚

```bash
python tools/run_subjective_ab.py \
  --model baseline::/home/ubuntu/model/netG_epoch_199.pth \
  --model candidate::/home/ubuntu/model/netG_finetuned_best.pth \
  --model rollback::/home/ubuntu/model/netG_epoch_199.pth \
  --output-json attack_results/subjective_ab_switch_log.json
```

该脚本会按顺序切换模型并记录每次 `/api/health` 返回，便于截图留档。

### 4) 报告材料一键打包

```bash
python tools/package_report_artifacts.py \
  --output-dir report_pack \
  --summary-md report_pack/REPORT_SUMMARY.md
```

输出：
- `report_pack/artifact_manifest.json`
- `report_pack/REPORT_SUMMARY.md`
