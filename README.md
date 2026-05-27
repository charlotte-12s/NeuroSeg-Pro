<div align="center">

# NeuroSeg Pro

**基于 LiteMedSAM 的专业医学图像交互式分割系统**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-%E2%89%A5%203.8-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)

[快速开始](#-快速开始) · [功能特性](#-功能特性) · [工作原理](#-工作原理) · [安装](#-安装) · [使用](#-使用) · [项目结构](#-项目结构)

</div>

---

## 一键点分割，秒级出结果

NeuroSeg Pro 将 Segment Anything 模型的强大能力带入医学影像领域。上传 NIfTI 图像，**点击病灶区域，即刻获取精准分割**——无需手动勾画，无需复杂参数。

---

## 界面预览

<table align="center">
  <tr>
    <td align="center"><b>Auto Mode — 自动分割</b></td>
    <td align="center"><b>Prompt Mode — 点击分割</b></td>
  </tr>
  <tr>
    <td><img src="test/auto_mode.png" alt="Auto Mode" width="480"/></td>
    <td><img src="test/prompt_mode.png" alt="Prompt Mode" width="480"/></td>
  </tr>
</table>

---

## 功能特性

| | | |
|:---:|:---:|:---:|
| 🖱️ **交互式点击分割** | 🧠 **3D NIfTI 支持** | 📊 **实时分析报告** |
| 点击病灶区域即可获取分割结果<br>零学习成本，直觉式操作 | 支持 2D / 3D NIfTI 格式<br>（.nii / .nii.gz）<br>3D 图像自动提取中间切片 | IoU 评估 + 病灶分析<br>量化分割质量，辅助临床决策 |
| | | |
| 🤖 **SAM 双模式** | ⚡ **轻量级推理** | 🔒 **本地化部署** |
| 自动分割 & Prompt 分割<br>支持点/框两种交互方式 | 基于 LiteMedSAM<br>更小的模型，更快的速度 | 数据不离开本机<br>符合医学影像隐私要求 |

---

## 工作原理

```
 ┌──────────────┐     ┌────────────────┐     ┌────────────────┐     ┌─────────────────────┐
 │              │     │                │     │                │     │                     │
 │  NIfTI 输入  │────▶│   预处理       │────▶│  SAM 推理      │────▶│  分割 + 分析报告    │
 │  .nii.gz     │     │  切片提取      │     │  LiteMedSAM    │     │  红色叠加 + IoU     │
 │              │     │  强度归一化     │     │  点/框提示      │     │  病灶量化           │
 └──────────────┘     └────────────────┘     └────────────────┘     └─────────────────────┘
```

1. **NIfTI 输入** — 上传 2D 或 3D 医学图像，系统自动识别维度与格式
2. **预处理** — 3D 图像自动提取中间切片，强度归一化至 SAM 兼容范围
3. **SAM 推理** — LiteMedSAM 编码器提取图像特征，结合用户点击/框选提示，解码器生成分割掩码
4. **分割 + 分析报告** — 红色叠加显示分割区域，同步输出 IoU 评估与病灶量化指标

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/charlotte-12s/NeuroSeg-Pro.git
cd NeuroSeg-Pro

# 2. 安装 Segment Anything
cd segment-anything-main && pip install -e . && cd ..

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载模型权重
mkdir -p weights
# 将 medsam_lite_best.pth 放入 weights/ 目录

# 5. 启动应用
python app.py
```

启动后访问 **http://localhost:7860**

> 权重文件需自行获取，不包含在本仓库中。

---

## 安装

<details>
<summary><b>详细安装步骤</b></summary>

### 1. 克隆仓库

```bash
git clone https://github.com/charlotte-12s/NeuroSeg-Pro.git
cd NeuroSeg-Pro
```

### 2. 安装 Segment Anything

```bash
cd segment-anything-main
pip install -e .
cd ..
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载模型权重

将 LiteMedSAM 权重文件放入 `weights/` 目录：

```bash
mkdir -p weights
# 将 medsam_lite_best.pth 放入 weights/ 目录
```

> 权重文件需自行获取，不包含在本仓库中。

### 依赖列表

| 依赖 | 说明 |
|------|------|
| Python >= 3.8 | 运行环境 |
| PyTorch | 深度学习框架 |
| Gradio | Web UI 框架 |
| MONAI | 医学图像处理 |
| OpenCV | 图像操作 |
| NumPy | 数值计算 |

</details>

---

## 使用

### 操作步骤

1. **上传图像** — 上传 NIfTI 图像文件（.nii.gz 或 .nii）
2. **自动预处理** — 系统自动预处理并显示图像
3. **点击目标** — 在图像上点击目标区域
4. **查看结果** — 查看分割结果（红色叠加区域）与分析报告

### 启动命令

```bash
python app.py
```

启动后访问 `http://localhost:7860`。

---

## 项目结构

<details>
<summary><b>展开查看完整项目结构</b></summary>

```
NeuroSeg-Pro/
├── app.py                  # Gradio Web UI 主程序
├── inference.py            # MedSAM 推理逻辑
├── build_model.py          # 模型构建（MedSAM_Lite / MedSAM_Lite_scribble）
├── models/                 # 自定义模型组件
│   ├── ImageEncoder/       # 图像编码器（TinyViT / ViT）
│   ├── common/             # 公共组件（MaskDecoder, LoRA, Adapter）
│   ├── efficient_sam/      # EfficientSAM 实现
│   └── efficientvit/       # EfficientViT 实现
├── segment-anything-main/  # Segment Anything 官方代码
├── weights/                # 模型权重（需自行放置）
├── examples/               # 示例图像
└── test/                   # 界面截图
```

</details>

---

## 致谢

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) — Meta AI
- [LiteMedSAM](https://github.com/bowang-lab/MedSAM) — MedSAM 团队
- [gradio-image-prompter](https://github.com/PhyscalX/gradio-image-prompter)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=charlotte-12s/NeuroSeg-Pro&type=Date)](https://star-history.com/#charlotte-12s/NeuroSeg-Pro&Date)

---

<div align="center">

**如果 NeuroSeg Pro 对你有帮助，给个 Star ⭐**

[![Star](https://img.shields.io/badge/Star-NeuroSeg--Pro-yellow?style=for-the-badge&logo=github)](https://github.com/charlotte-12s/NeuroSeg-Pro)
[![Bug](https://img.shields.io/badge/Report-Bug-red?style=for-the-badge&logo=github)](https://github.com/charlotte-12s/NeuroSeg-Pro/issues)
[![Feature](https://img.shields.io/badge/Request-Feature-blue?style=for-the-badge&logo=github)](https://github.com/charlotte-12s/NeuroSeg-Pro/issues)

<sub>Released under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0)</sub>

</div>
