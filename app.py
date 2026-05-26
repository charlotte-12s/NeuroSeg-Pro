import os
# 设置环境变量避免 OpenMP 警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gradio as gr
import numpy as np
import cv2
from inference import MedSAMInference


# 全局初始化 MedSAMInference 实例（单例模式）
_inference_instance = None

def get_inference():
    """获取全局 MedSAMInference 实例（单例模式）"""
    global _inference_instance
    if _inference_instance is None:
        print("正在初始化 MedSAM Inference...")
        _inference_instance = MedSAMInference()
        print("MedSAM Inference 初始化完成！")
    return _inference_instance


def extract_middle_slice(image):
    """
    从 3D 图像中提取中间切片
    
    Args:
        image: numpy array，可能是 (C, H, W) 或 (C, H, W, D)
        
    Returns:
        numpy array: 2D 图像 (C, H, W)
    """
    if image.ndim == 4:
        # 3D 图像 (C, H, W, D)，提取中间切片
        depth = image.shape[3]
        middle_idx = depth // 2
        image_2d = image[:, :, :, middle_idx]
        print(f"检测到 3D 图像，提取中间切片 (索引: {middle_idx}/{depth-1})")
    elif image.ndim == 3:
        # 已经是 2D 图像 (C, H, W)
        image_2d = image
    else:
        raise ValueError(f"不支持的图像维度: {image.ndim}")
    
    return image_2d


def preprocess_image_to_display(image_2d):
    """
    将预处理后的图像转换为用于显示的格式 (uint8, 0-255)
    
    Args:
        image_2d: numpy array (C, H, W)，值范围 0-1
        
    Returns:
        numpy array: (H, W, 3) uint8，值范围 0-255，RGB 格式
    """
    # 移除通道维度，取第一个通道（如果是多通道）
    if image_2d.shape[0] > 1:
        image_2d = image_2d[0]  # 取第一个通道
    else:
        image_2d = image_2d.squeeze(0)  # (H, W)
    
    # 归一化到 0-255
    if image_2d.max() <= 1.0:
        image_2d = (image_2d * 255).astype(np.uint8)
    else:
        image_2d = image_2d.astype(np.uint8)
    
    # 转换为 RGB (如果是灰度图，复制为 3 通道)
    if len(image_2d.shape) == 2:
        image_rgb = cv2.cvtColor(image_2d, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_2d
    
    return image_rgb


def upload_and_preprocess(file):
    """
    上传并预处理 NIfTI 文件
    
    Args:
        file: Gradio File 对象
        
    Returns:
        tuple: (display_image, preprocessed_image_state)
            - display_image: 用于显示的图像 (H, W, 3) uint8
            - preprocessed_image_state: 预处理后的原始图像数据 (C, H, W) 用于后续推理
    """
    if file is None:
        return None, None
    
    try:
        # 获取文件路径
        # Gradio File 组件返回文件路径字符串或 FileData 对象
        if isinstance(file, str):
            file_path = file
        elif hasattr(file, 'name'):
            file_path = file.name
        else:
            file_path = str(file)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 获取推理实例并预处理
        inference = get_inference()
        preprocessed_image = inference.preprocess(file_path)
        
        # 提取中间切片（如果是 3D）
        image_2d = extract_middle_slice(preprocessed_image)
        
        # 转换为显示格式
        display_image = preprocess_image_to_display(image_2d)
        
        print(f"图像预处理完成，形状: {image_2d.shape}, 显示图像形状: {display_image.shape}")
        
        return display_image, image_2d
        
    except Exception as e:
        print(f"预处理错误: {str(e)}")
        raise gr.Error(f"图像预处理失败: {str(e)}")


def on_image_select(evt: gr.SelectData, preprocessed_image_state):
    """
    处理图像点击事件，进行分割预测
    
    Args:
        evt: Gradio SelectData 事件对象
        preprocessed_image_state: 预处理后的图像数据 (C, H, W)
        
    Returns:
        numpy array: 叠加了红色半透明 mask 的图像 (H, W, 3) uint8
    """
    if preprocessed_image_state is None:
        raise gr.Error("请先上传 NIfTI 图像！")
    
    try:
        # 获取点击坐标
        # evt.index 是 (row, col)，即 (y, x)
        click_y, click_x = evt.index[0], evt.index[1]
        
        # 注意：Gradio 的坐标是 (y, x)，但模型需要 (x, y)
        # 同时，Gradio 显示的图像可能是原始尺寸，但模型输入是 256x256
        # 需要将点击坐标映射到 256x256 空间
        
        # 获取显示图像的尺寸（用于坐标映射）
        display_image = preprocess_image_to_display(preprocessed_image_state)
        display_h, display_w = display_image.shape[:2]
        
        # 模型输入尺寸是 256x256
        model_size = 256
        
        # 将点击坐标映射到模型输入空间
        x_scaled = int(click_x * model_size / display_w)
        y_scaled = int(click_y * model_size / display_h)
        
        # 确保坐标在有效范围内（防止 Gradio 拉伸导致坐标越界）
        x_scaled = max(0, min(255, x_scaled))
        y_scaled = max(0, min(255, y_scaled))
        
        print(f"点击坐标: ({click_x}, {click_y}) -> 模型坐标: ({x_scaled}, {y_scaled})")
        
        # 获取推理实例并进行预测
        inference = get_inference()
        mask, iou = inference.predict(preprocessed_image_state, [x_scaled, y_scaled])
        
        print(f"分割完成，IoU: {iou:.4f}")
        
        # 将 mask 调整到显示图像尺寸
        if mask.shape != (display_h, display_w):
            mask_resized = cv2.resize(mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        
        # 确保 mask 是二值的 (0 或 1)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # 创建红色 mask 叠加层
        # 注意：OpenCV 使用 BGR 格式，红色是 [0, 0, 255]
        red_mask = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        red_mask[:, :, 2] = 255  # 红色通道 (BGR 格式)
        
        # 先对整个图像进行 cv2.addWeighted 混合，得到混合后的图像
        # result = image * (1 - alpha) + mask * alpha
        blended = cv2.addWeighted(display_image, 0.5, red_mask, 0.5, 0)
        
        # 创建 overlay，只在 mask 区域使用混合后的颜色
        overlay = display_image.copy()
        mask_area = mask_binary > 0  # mask 区域
        if np.any(mask_area):
            # 将混合后的颜色赋值给 mask 区域
            overlay[mask_area] = blended[mask_area]
        
        # 生成病灶分析报告
        mask_pixels = np.sum(mask_binary > 0)
        total_pixels = mask_binary.size
        mask_percentage = (mask_pixels / total_pixels) * 100
        
        report = f"""✅ 分割完成

📊 分析结果：
• IoU 分数: {iou:.4f}
• 病灶区域像素数: {mask_pixels:,}
• 病灶占比: {mask_percentage:.2f}%
• 点击坐标: ({click_x}, {click_y})
• 模型坐标: ({x_scaled}, {y_scaled})

💡 提示：红色区域表示分割出的病灶区域"""
        
        return overlay, report
        
    except Exception as e:
        print(f"分割预测错误: {str(e)}")
        raise gr.Error(f"分割预测失败: {str(e)}")


def clear_all():
    """清除所有输入和输出"""
    return None, None, None, "等待上传图像..."


# 自定义 CSS 样式
custom_css = """
/* 隐藏页脚 */
footer {display: none !important;}

/* 头部 Banner 样式 */
.header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    color: white;
    text-align: center;
}

.header-banner h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.header-banner p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.95;
}

/* 图片容器卡片样式 */
.image-card {
    border-radius: 12px;
    padding: 1rem;
    background: white;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}

.image-card:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}

/* 美化按钮 */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.secondary-btn {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.secondary-btn:hover {
    background: #e2e8f0;
    transform: translateY(-1px);
}

/* 报告文本框样式 */
.report-box {
    border-radius: 8px;
    border: 2px solid #e2e8f0;
    background: #f8fafc;
    font-family: 'Courier New', monospace;
    line-height: 1.6;
}

/* Accordion 样式 */
.accordion-content {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
}
"""

# 创建 Gradio 界面
title = "🧠 NeuroSeg Pro - 专业医学图像分割工作站"

with gr.Blocks(title=title) as demo:
    # 添加自定义 CSS
    gr.HTML(f"""
    <style>
    {custom_css}
    </style>
    """)
    
    # 头部 Banner
    gr.HTML("""
    <div class="header-banner">
        <h1>🧠 NeuroSeg Pro</h1>
        <p>基于 LiteMedSAM 的专业医学图像交互式分割系统</p>
    </div>
    """)
    
    # 使用 State 保存预处理后的图像数据
    preprocessed_image_state = gr.State(value=None)

    with gr.Row():
        # 左侧：输入区域
        with gr.Column(scale=1):
            # 图片输入容器
            with gr.Group(elem_classes="image-card"):
                input_image = gr.Image(
                    label="📥 输入图像",
                    type="numpy",
                                interactive=True,
                    height=500,
                        show_label=True
                )
            
            # 文件上传组件
            file_upload = gr.File(
                label="📁 上传 NIfTI 文件 (.nii.gz)",
                file_types=[".nii.gz", ".nii"],
                height=80
            )
            
            # 操作指南（Accordion）
            with gr.Accordion("📖 操作指南", open=False):
                gr.Markdown("""
                ### 使用步骤：
                1. **上传图像**：点击上方"上传 NIfTI 文件"按钮，选择您的医学图像文件（.nii.gz 或 .nii 格式）
                2. **查看图像**：上传后，系统会自动预处理并显示图像（如果是 3D 图像，会提取中间切片）
                3. **点击分割**：在左侧图像上点击目标区域（病灶位置），系统会自动进行分割
                4. **查看结果**：右侧会显示分割结果，红色区域表示分割出的病灶
                5. **分析报告**：每次分割后，下方会显示详细的分析报告，包括 IoU 分数、病灶占比等信息
                
                ### 注意事项：
                - 支持 2D 和 3D NIfTI 格式图像
                - 点击位置应尽量准确，建议点击病灶中心区域
                - 分割结果以红色半透明叠加显示
                """)
        
        # 右侧：输出区域
        with gr.Column(scale=1):
            # 分割结果容器
            with gr.Group(elem_classes="image-card"):
                output_image = gr.Image(
                    label="📤 分割结果",
                    type="numpy",
                            interactive=False, 
                    height=500,
                    show_label=True
                )
            
            # 病灶分析报告
            report_text = gr.Textbox(
                label="📋 病灶分析报告",
                value="等待上传图像...",
                lines=8,
                max_lines=12,
                            interactive=False, 
                elem_classes="report-box"
            )
            
            # 清除按钮
            clear_btn = gr.Button(
                "🗑️ 清除所有",
                variant="secondary",
                size="lg",
                elem_classes="secondary-btn"
            )
    
    # 事件绑定
    # 文件上传时预处理图像
    file_upload.upload(
        fn=upload_and_preprocess,
        inputs=[file_upload],
        outputs=[input_image, preprocessed_image_state]
    )
    
    # 点击图像时进行分割
    input_image.select(
        fn=on_image_select,
        inputs=[preprocessed_image_state],
        outputs=[output_image, report_text]
    )
    
    # Clear 按钮
    clear_btn.click(
        fn=clear_all,
        outputs=[input_image, output_image, preprocessed_image_state, report_text]
    )


if __name__ == "__main__":
    # 启动应用
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,  # 端口号
        share=False  # 是否创建公共链接
    )
