# Krea 2 LoRA 训练 on 2080Ti（Turing）——九版踩坑全记录与最终工作配方

> 2026-11-07 定稿。九个版本、三套框架（DiffSynth / OmniChar×2）失败后定位到根因并跑通。
> 本文是唯一权威记录：脚本索引、根因证据、工作配方、判定仪器、校准知识都在这里。

## 0. TL;DR

- **根因：fp16 反向传播梯度下溢。** GRADLOG 实测 **99.99% 的 LoRA 梯度分量幅值 < fp16 最小正规数 6.1e-5**，裸 fp16 反向把它们冲成零/亚正规噪声——身份/条件相关的小分量信号从未到达优化器。v1–v8 全部死于此。
- **修复三件套（v9，一次全上）后角色首次出现：**
  1. **GradScaler（loss×65536 反向，事后除回）**——修复下溢，验证后 zero-frac 0.00002
  2. **LoRA 目标砍到 attention+ff**（7 类模块；弃 text_fusion/img_in/final_layer/txt_in——梯度下降会先劫持这些便宜的全局旋钮学画风，饿死身份）
  3. **sigma 采样改 shift 2.5**（`2.5u/(1+1.5u)`，u~U(0,1)；对齐 musubi 验证配方与推理侧噪声分布）
- 佐证：训练 loss 首次低于底模裸跑（**0.0590 < 0.0599**）；loss 比值 0.77（此前所有版本 ≥1.0）。

## 1. 硬件与精度背景（为什么这张卡上全是坑）

2080Ti = Turing sm75：
- fp16 张量核 ✅（快） | **bf16 ❌**（模拟，实测 ~5× 慢：21.7 s/步 vs 4.3）| fp8 ❌ | Triton convrot int8 编译失败
- fp16 动态范围两端都要命，是同一个病的两张脸：
  - **上溢端（NaN）**：>65504 → inf → softmax NaN。实测两处爆点：
    - QK-norm 可学习 scale（max 51.9）→ q·k 点积放大 ~2700× → **修法：注意力内部升 fp32 计算**（进出仍 fp16）
    - 残差流纵向累积（block23–27 常态 12–20k = 30% 天花板；block7 特定样本尖峰 52.5k→inf）→ **修法：块间残差骨架 fp32，块内 matmul 保持 fp16**。速度损失 ≈0（5.0 vs 5.5 s/步）
  - **下溢端（本记录的真凶）**：<6.1e-5 → 0/亚正规。GradScaler 修。**注意：GradScaler 只在「正向不溢出」的前提下有效**——所以上溢端的两块护甲是它的前置条件。
- LoRA 主权重强制 fp32（PEFT 默认继承底模 dtype=fp16，配 8bit Adam 是灾难）
- 梯度裁剪 1.0（AI-Toolkit/musubi 标准；无裁剪时 loss 0.098→0.23 暴涨形态实测）

## 2. 九版失败链（每一版的假设与死因）

| 版本 | 配置 | 结果 | 死因/教训 |
|---|---|---|---|
| v1 | OmniChar 全默认：lr1e-4 batch1 fp16 8bitAdam 全目标 | 1500 步西装男，block0 max 4.4 | 当时归因 lr 太大；真凶是下溢+目标集过宽 |
| DiffSynth 版 | bf16 模拟 | 22.5 s/步 | 太慢，放弃（也测出 1024 在 22G 放不下） |
| v3 | +fp32 主权重 +裁剪 | step 618 NaN | 上溢端：注意力 QK-norm → 补 fp32 注意力 |
| v5 | +fp32 注意力, rank32 2500步 | step ~760 雪崩 | 残差流 block7 溢出 → 补 fp32 残差骨架（ff 实验 5e-4 下精确定位 223 步爆点） |
| v6 | +fp32 残差 +fp32 AdamW | 1300 步后 NaN；权重干净但只有目标 1/5 大小 | 上溢全堵住后暴露真瓶颈：**adapter 长不出身份** |
| v7 | 数据手术：123 张紧裁块 | loss 探针 -8.5% 仍西装男；触发词零绑定 | 门禁被 prompt 污染（helmet 词），纯触发词测试证死 |
| v8 | 社区原样 lr1e-4 batch1 | 600 步内 @0.35 零效果 / @1.0 噪点；loss 升 | **噪点@1.0 = 病理**（用户指出正常 LoRA 任何步数 @1.0 都不应解体）→ 引出下溢怀疑 |
| ff 系列 | lr 5e-4 快速失败 + 全模块埋点 | 223 步 block7 52.5k→inf | 爆点定位仪；fp32 残差后 400/400 步 0 溢出 |
| **v9** | **三件套** | **角色出现**；loss 0.0590 < 底模 0.0599 | ✅ |

## 3. 最终工作配方（v9）

```bash
# launcher（含全部补丁：窄目标集 / shift sigma / fp32 注意力 / fp32 残差 / GRADLOG）
scripts/omni_v9_narrow_shift.py + scripts/omni_manifest_v9.json
# 启动
cd ~/train/omni && setsid nohup env PYTHONPATH=~/train/omni_libs/OmniChar/core/src \
  HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/train/venv/bin/python scripts/omni_v9_narrow_shift.py scripts/omni_manifest_v9.json \
  > logs/omni_v9.log 2>&1 < /dev/null &
```

关键超参：lr 1e-4 / batch 1 / rank 32 α32 / 512px / 123 图裁块集 / 每 100 步快照。
速度 ~5.5 s/步，显存 13.6G。trainer 持久副本已打 scaler+GRADLOG+裁剪补丁（`scripts/patch_scaler.py` 可复放）。
数据集：`dataset_crop/` = 26 张全身×2 错位裁 + 16 张并排全身 + 15 张特写中心裁；caption = `q21char, <部位/景别>`（无白底词）。

## 4. 判定仪器（probes/，按价值排序）

1. **grad_probe.py** — fp16 / bf16 / fp16+scaler 三臂同批反向，梯度零值率+余弦对比。**新卡/新栈上来先跑这个。**
2. **纯触发词测试** — 提示词只剩 `q21char`，任何装备特征只能来自 LoRA。判定金标准（防提示词污染）。
3. **text_binding 探针** — 同 latent 下 caption vs 空 embed 的 loss 差。底模 ~3.6%；adapter 归零=条件通路被压死。
4. **loss_probe.py** — adapter vs 底模在训练集上的拟合。低于底模 = 真学习（v9 首次达成）。
5. **强度轴采样 + 同 seed 无 LoRA 对照** — @0.35/0.5/0.8/1.0。噪点@1.0 = 病理信号。
6. **GRADLOG（训练内置）** — zero-frac / would-underflow-frac / max 每步打印，下溢实时可见。

## 5. 校准知识（判读标准）

- **健康演化**：0–500 步全局外观渗出（色板/发型/装备类型）→ 500–1500 特征固化 → 1500+ 打磨/过拟合。
- **两个死亡形态要区分**：全强度零变化=没学；有变化但换人/纯画风=学错方向（无条件偏移）。
- **@1.0 噪点 = 病理**，不是"训多了"。正常 LoRA @1.0 是设计工作点。
- **提示词行为（cfg=1 turbo）**：纯触发词 > 触发词+描述词——描述词（standing/full body/white background）激活底模先验（西装男模式）并稀释触发词权重，无反向力纠正。**要相似度：触发词主导、场景词最少；要场景：接受折损或提强度补偿。**
- **门禁设计教训**：提示词混淆变量咬了我们两次（helmet 词）。判定一律用干净提示词 + 无 LoRA 对照。

## 6. 未了事项

- [ ] v9x：从 ckpt_v9（600 步）续到 2500 步完整训练（manifest `omni_manifest_v9x.json` 已就绪，~2.9h）
- [ ] 强度扫描定操作配方（预期 0.8–1.2 区间）
- [ ] 场景图实测（触发词+最少场景词；对照"纯触发词最大相似度"基线）
- [ ] heal LoRA 效果评估（沙发图已出未系统对比）

## 7. 参考

- 验证可用的社区配方：note.com sepiablue《【VRAM 12GB】Krea2 キャラ LoRA 完全ガイド（musubi-tuner）》——256px/20图/2000步成功；`--timestep_sampling shift --discrete_flow_shift 2.5`；`networks.lora_krea2` 264 目标；RAW 训练→Turbo 推理。4070 原生 bf16 2 小时跑量 ≈ 我们 4–8 小时（其 Ada 算力 ~2×），故留在自栈。
