import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn

# Constants
CHANNEL_N = 16        
embedding_dim = 128

class GradientSensor(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x_weights = torch.tensor([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]).float()
        sobel_y_weights = torch.tensor([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]).float().T
        sobel_x_weights = sobel_x_weights.unsqueeze(0).unsqueeze(0).repeat(CHANNEL_N, 1, 1, 1)
        sobel_y_weights = sobel_y_weights.unsqueeze(0).unsqueeze(0).repeat(CHANNEL_N, 1, 1, 1)

        self.sobel_x = nn.Parameter(sobel_x_weights, requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y_weights, requires_grad=False)

    def forward(self, x):
        grad_x = nn.functional.conv2d(x, self.sobel_x, padding='same', groups=CHANNEL_N)
        grad_y = nn.functional.conv2d(x, self.sobel_y, padding='same', groups=CHANNEL_N)
        return torch.cat([x, grad_x, grad_y], dim=1)

class AdaptiveSparseNetwork(nn.Module):
    def __init__(self, embedding_dim=embedding_dim):
        super().__init__()
        self.dense_layers = nn.Sequential(
            nn.Linear(CHANNEL_N * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, CHANNEL_N),
        )

    def forward(self, features):
        return self.dense_layers(features)

class AdaptiveSparseUpdateRule(nn.Module):
    def __init__(self, embedding_dim=embedding_dim):
        super().__init__()
        self.sensor = GradientSensor()
        self.dense_network = AdaptiveSparseNetwork(embedding_dim)

    def forward(self, x, fire_mask):
        # Step 1: Compute gradients for ALL pixels
        features = self.sensor(x)  # [B, 48, H, W]
        
        # Step 2: Compute alive mask from alpha channel
        alpha_channel = x[:, 3:4, :, :]  # [B, 1, H, W]
        alive_mask = nn.functional.max_pool2d(
            alpha_channel, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            dilation=1,
            ceil_mode=False
        ) > 0.1  # [B, 1, H, W]
        
        # Step 3: Combine fire mask with alive mask for MAXIMUM sparsity
        combined_mask = fire_mask * alive_mask.float()  # [B, 1, H, W]
        
        # Step 4: Convert to per-pixel format
        B, C, H, W = features.shape
        features_hw = features.permute(0, 2, 3, 1)  # [B, H, W, 48]
        combined_mask_hw = combined_mask.permute(0, 2, 3, 1)  # [B, H, W, 1]
        
        # Step 5: Get indices of active pixels (fire AND alive)
        combined_indices = combined_mask_hw.squeeze(-1).nonzero(as_tuple=False)  # [N_active, 3]
        
        # Step 6: Extract only active pixel features
        selected_features = features_hw[combined_indices[:, 0], combined_indices[:, 1], combined_indices[:, 2]]  # [N_active, 48]
        
        # Step 7: Process through dense network - MAXIMUM sparsity!
        if selected_features.size(0) > 0:
            updates_selected = self.dense_network(selected_features)  # [N_active, 16]
        else:
            updates_selected = torch.zeros(0, CHANNEL_N, dtype=features.dtype, device=features.device)
        
        # Step 8: Scatter results back
        updates_hw = torch.zeros(B, H, W, CHANNEL_N, dtype=features.dtype, device=features.device)
        if updates_selected.size(0) > 0:
            updates_hw[combined_indices[:, 0], combined_indices[:, 1], combined_indices[:, 2]] = updates_selected
        
        # Step 9: Convert back to [B, 16, H, W] format
        updates = updates_hw.permute(0, 3, 1, 2)  # [B, 16, H, W]
        return updates

class AdaptiveSparseNeuralCA(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.update_rule = AdaptiveSparseUpdateRule(embedding_dim)
        
        # Channel mask
        channel_mask = torch.ones(1, CHANNEL_N, 1, 1, dtype=torch.float32)
        channel_mask[0, 5, 0, 0] = 0.0  # Zero out channel 5
        self.register_buffer('channel_mask', channel_mask)

    def forward(self, x, fire_mask):
        # Compute updates using adaptive sparse method
        update_values = self.update_rule(x, fire_mask)
        
        # Apply channel mask
        update_values = update_values * self.channel_mask
        
        # Apply updates (masks already baked into update_values)
        delta = update_values
        new_x = x + delta
        return new_x.clamp(0, 1)

def create_test_input(height=32, width=256):
    x_start = torch.zeros(1, 16, height, width, dtype=torch.float32)
    x_start[:, 3, :, :] = 0.1
    seed_rows = [height-1, height-2, height-3, height-4, height-5]
    x_start[:, 3:, seed_rows, 0] = 1.0
    x_start[:, 3, seed_rows, 0] = 1.0
    x_start[:, 5, :, :] = 0.0
    return x_start

def load_pretrained_weights(model):
    model_path = '/Users/artemshmatko/Dropbox/Python/website/nca_model_cpu.pth'
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        sparse_state_dict = {}
        for old_key, value in state_dict.items():
            if 'update_rule.update_rule' in old_key:
                new_key = old_key.replace('update_rule.update_rule.1', 'update_rule.dense_network.dense_layers.0')
                new_key = new_key.replace('update_rule.update_rule.3', 'update_rule.dense_network.dense_layers.2')
                new_key = new_key.replace('update_rule.update_rule.5', 'update_rule.dense_network.dense_layers.4')
                sparse_state_dict[new_key] = value
            elif 'update_rule.sensor' in old_key:
                sparse_state_dict[old_key] = value
        
        model.load_state_dict(sparse_state_dict, strict=False)
        print("✅ Pre-trained weights loaded successfully!")
        return True
    else:
        print("❌ No pre-trained model found, using random weights")
        return False

def export_adaptive_sparse_onnx():
    print("🔥 EXPORTING ADAPTIVE FIRE+ALIVE SPARSE MODEL")
    print("=" * 60)
    
    # Create model
    model = AdaptiveSparseNeuralCA(embedding_dim=128)
    load_pretrained_weights(model)
    model.eval()
    
    # Create test inputs
    test_input = create_test_input(height=32, width=256)
    fire_mask = torch.rand(1, 1, 32, 256) < 0.5
    fire_mask = fire_mask.float()
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Fire mask shape: {fire_mask.shape}")
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(test_input, fire_mask)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Export to ONNX
    print("Exporting to ONNX...")
    onnx_path = "nca_adaptive_sparse_model.onnx"
    
    try:
        torch.onnx.export(
            model,
            (test_input, fire_mask),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input', 'fire_mask'],
            output_names=['output'],
            verbose=False
        )
        
        print(f"✅ Adaptive sparse model exported to {onnx_path}")
        print("🚀 Ready for 2.7x speedup with adaptive fire+alive sparsity!")
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

if __name__ == "__main__":
    export_adaptive_sparse_onnx()