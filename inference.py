import os
# 设置环境变量避免 OpenMP 警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from monai.transforms import LoadImage, EnsureChannelFirst, Orientation, ScaleIntensity, Compose
from build_model import build_model


class Args:
    """简单的参数类，用于构建模型"""
    def __init__(self):
        self.image_size = 256  # LiteMedSAM 输入尺寸固定为 256x256
        self.encoder_adapter = True  # 使用编码器适配器
        self.mod = 'sam_adpt'  # MedSAM Lite 常用的适配器模式
        self.mid_dim = 64  # Adapter 中间维度，必须与 checkpoint 训练时的维度一致
        self.compile = False  # 编译选项
        self.verbose = False  # 详细输出选项
        self.func = None  # 功能选项
        self.thd = False  # 是否使用 3D 分支，False 表示使用 2D 分割
        self.chunk = None  # 3D 分支的深度块大小
        self.num_classes = 1  # 分割类别数


class MedSAMInference:
    """MedSAM Lite 推理类，用于医学图像分割"""
    
    def __init__(self, device=None):
        """
        初始化推理类
        
        Args:
            device: 指定设备 ('cuda' 或 'cpu')，如果为 None 则自动检测
        """
        # 检测设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"使用设备: {self.device}")
        
        # 构造 dummy args
        args = Args()
        args.image_size = 256  # LiteMedSAM 输入尺寸固定为 256x256
        
        # 构建模型
        print("正在构建模型...")
        self.model = build_model(args, is_boxes=True)
        self.model.to(self.device)
        
        # 加载权重
        weight_path = "./weights/medsam_lite_best.pth"
        print(f"正在加载权重: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=self.device)
        
        # 处理 state_dict：检查是否有 "module." 前缀（多卡训练遗留）
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            # 如果 checkpoint 是字典，尝试获取 'state_dict' 或 'model' 键
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        
        # 去除 "module." 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key[7:]  # 去除 "module." 前缀
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 加载权重
        self.model.load_state_dict(new_state_dict, strict=True)
        print("权重加载完成")
        
        # 设置为评估模式
        self.model.eval()
        
        # 定义预处理 transform 链
        # 注意：不在这里进行 Resize，因为输入可能是 3D 图像，Resize 会在提取切片后单独处理
        self.transform = Compose([
            LoadImage(image_only=True),  # 加载图像
            EnsureChannelFirst(),  # 确保通道维度在前
            Orientation(axcodes="RAS"),  # 转换为 RAS 方向
            ScaleIntensity(minv=0.0, maxv=1.0)  # Min-Max 归一化到 0-1 范围（适用于 MRI 数据，如 BraTS）
        ])
    
    def preprocess(self, nii_path):
        """
        预处理医学图像（NIfTI 格式）
        
        Args:
            nii_path: NIfTI 图像路径
            
        Returns:
            numpy.ndarray: 预处理后的图像，可能是 (C, H, W) 或 (C, H, W, D)
                          注意：不包含 Resize，因为输入可能是 3D 图像
        """
        # 应用 transform
        image = self.transform(nii_path)
        
        # 转换为 numpy
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        return image
    
    def predict(self, image_2d, prompt_coords):
        """
        对单张 2D 图像进行分割预测
        
        Args:
            image_2d: 预处理后的图像，可以是 numpy array (C, H, W) 或 tensor
            prompt_coords: 提示点坐标，格式为 [x, y]（图像坐标系）
            
        Returns:
            tuple: (mask, iou)
                - mask: 分割掩码，numpy array，形状为 (H, W)，值为 0 或 1
                - iou: IoU 预测值，float
        """
        # 确保输入是 tensor
        if isinstance(image_2d, np.ndarray):
            image_tensor = torch.from_numpy(image_2d).float()
        else:
            image_tensor = image_2d.float()
        
        # 确保图像是单通道，需要转换为 3 通道（LiteMedSAM 需要 3 通道输入）
        if image_tensor.dim() == 2:
            # (H, W) -> (1, H, W)
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.shape[0] == 1:
            # (1, H, W) -> (3, H, W) 通过复制通道
            image_tensor = image_tensor.repeat(3, 1, 1)
        
        # 注意：不要添加 batch 维度，因为 build_model.py 里的 torch.stack 会自动添加
        # image 的形状应该是 (3, 256, 256)
        
        # 确保尺寸为 256x256
        if image_tensor.shape[1] != 256 or image_tensor.shape[2] != 256:
            # 需要先添加 batch 维度进行插值，然后移除
            image_tensor_batch = image_tensor.unsqueeze(0)
            image_tensor_batch = torch.nn.functional.interpolate(
                image_tensor_batch, 
                size=(256, 256), 
                mode='bilinear', 
                align_corners=False
            )
            image_tensor = image_tensor_batch.squeeze(0)
        
        # 移动到设备
        image_tensor = image_tensor.to(self.device)
        
        # 构造 batched_input
        # point_coords 需要是 (1, 1, 2) 的形状 (Batch=1, Num_points=1, XY=2)
        point_coords_tensor = torch.tensor(
            [[prompt_coords]], 
            dtype=torch.float, 
            device=self.device
        )  # Shape: (1, 1, 2) - Batch=1, 1 个点，2 个坐标 (x, y)
        
        point_labels_tensor = torch.tensor(
            [[1]], 
            dtype=torch.float, 
            device=self.device
        )  # Shape: (1, 1) - Batch=1, 1 代表前景点
        
        batched_input = [{
            'image': image_tensor,
            'point_coords': point_coords_tensor,
            'point_labels': point_labels_tensor,
            'original_size': (256, 256)  # 输入给 encoder 的尺寸
        }]
        
        # 推理
        with torch.no_grad():
            low_res_masks, iou = self.model(batched_input, multimask_output=False)
        
        # 后处理：应用 sigmoid 并转换为二值 mask
        # low_res_masks 形状可能是 (B, C, H, W) 或 (B, C, 1, H, W)，使用 squeeze 去除所有为 1 的维度
        mask_logits = low_res_masks.squeeze()  # 去除所有为 1 的维度
        
        # 确保得到 2D 的 (H, W) 矩阵
        while mask_logits.dim() > 2:
            mask_logits = mask_logits[0]  # 如果维度仍大于 2，取第一个元素
        
        mask_probs = torch.sigmoid(mask_logits)  # 应用 sigmoid
        mask_binary = (mask_probs > 0.5).float()  # 阈值化
        
        # 转换为 numpy，确保是 2D
        mask_np = mask_binary.cpu().numpy().astype(np.uint8)
        mask_np = np.squeeze(mask_np)  # 去除所有为 1 的维度
        
        # 最终确保是 2D 数组，如果不是则强制 reshape
        if mask_np.ndim == 1:
            # 如果是一维，reshape 为 2D（这种情况不应该发生）
            mask_np = mask_np.reshape(256, 256)
        elif mask_np.ndim > 2:
            # 如果是多维，取第一个 2D slice
            while mask_np.ndim > 2:
                mask_np = mask_np[0]
        
        # 最终验证：确保是 2D
        assert mask_np.ndim == 2, f"mask_np 应该是 2D，但得到 {mask_np.ndim}D，形状: {mask_np.shape}"
        
        iou_value = iou.squeeze().cpu().item()  # 取出 IoU 值
        
        return mask_np, iou_value


if __name__ == "__main__":
    # 示例用法
    print("初始化 MedSAM Inference...")
    inference = MedSAMInference()
    
    # 示例：预处理图像
    # nii_path = "path/to/your/image.nii.gz"
    # preprocessed_image = inference.preprocess(nii_path)
    
    # 示例：预测
    # prompt_coords = [128, 128]  # 图像中心点
    # mask, iou = inference.predict(preprocessed_image, prompt_coords)
    # print(f"预测完成，IoU: {iou:.4f}")
    print("MedSAM Inference 初始化完成！")

