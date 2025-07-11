import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def test_original_preprocessing(image_path, model_path):
    """Test with the original [0,1] preprocessing that gave banded results"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    
    # ORIGINAL preprocessing (that gave banded results)
    image = Image.open(image_path).convert('L').resize((256, 256))
    
    # Original approach that gave horizontal bands
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts [0-255] → [0-1]
    ])
    
    tensor = transform(image).unsqueeze(0).to(device)
    print(f"Original preprocessing - tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
    
    # Create data
    data = {
        'tgt_image': tensor,
        'tgt_depth_gt': torch.zeros_like(tensor)
    }
    
    with torch.no_grad():
        model.feed_data(data)
        model.test(continuous=False)
        visuals = model.get_current_visuals()
        predicted = visuals['Predicted']
        
        if isinstance(predicted, list):
            predicted = predicted[-1]
    
    pred_array = predicted.squeeze().cpu().numpy()
    
    print(f"Result with original preprocessing:")
    print(f"   Range: [{pred_array.min():.4f}, {pred_array.max():.4f}]")
    print(f"   Variance: {pred_array.var():.6f}")
    
    # Check spatial correlation
    from scipy.stats import pearsonr
    flat_depth = pred_array.flatten()
    shifted = np.roll(flat_depth, 1)
    correlation, _ = pearsonr(flat_depth[1:], shifted[1:])
    print(f"   Spatial correlation: {correlation:.4f}")
    
    if correlation > 0.3:
        print("Shows structure (horizontal bands)")
    else:
        print("Still random")
    
    # Save this result with GROUND TRUTH SIZE
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.dirname(image_path)
    
    # Get ground truth size
    original_size = Image.open(image_path).size
    print(f"   Resizing to ground truth size: {original_size}")
    
    # Normalize and save
    normalized = (pred_array - pred_array.min()) / (pred_array.max() - pred_array.min() + 1e-8)
    normalized_uint8 = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL and resize to ground truth size
    colored_image = Image.fromarray(colored_rgb)
    colored_resized = colored_image.resize(original_size, Image.LANCZOS)
    
    output_path = os.path.join(output_dir, f"{base_name}_original_preprocessing.png")
    colored_resized.save(output_path)
    print(f"   Saved: {output_path} (size: {colored_resized.size})")
    
    return pred_array

def test_intermediate_preprocessing(image_path, model_path):
    """Test with intermediate normalization values"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Try different scaling factors
    scaling_factors = [1.0, 2.0, 127.5, 255.0]  # Different ways to scale [0,1] back up
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.dirname(image_path)
    
    # Get original size
    original_size = Image.open(image_path).size
    print(f"Original image size: {original_size}")
    
    best_correlation = 0
    best_factor = None
    best_result = None
    
    for factor in scaling_factors:
        print(f"\nTesting scaling factor: {factor}")
        
        # Load image
        image = Image.open(image_path).convert('L').resize((256, 256))
        transform = transforms.Compose([transforms.ToTensor()])
        tensor = transform(image).unsqueeze(0).to(device)  # [0,1] range
        
        # Scale it up
        scaled_tensor = tensor * factor
        print(f"   Scaled tensor range: [{scaled_tensor.min():.1f}, {scaled_tensor.max():.1f}]")
        
        # Test with model
        data = {
            'tgt_image': scaled_tensor,
            'tgt_depth_gt': torch.zeros_like(scaled_tensor)
        }
        
        with torch.no_grad():
            model.feed_data(data)
            model.test(continuous=False)
            visuals = model.get_current_visuals()
            predicted = visuals['Predicted']
            
            if isinstance(predicted, list):
                predicted = predicted[-1]
        
        pred_array = predicted.squeeze().cpu().numpy()
        
        # Check spatial correlation
        from scipy.stats import pearsonr
        flat_depth = pred_array.flatten()
        shifted = np.roll(flat_depth, 1)
        correlation, _ = pearsonr(flat_depth[1:], shifted[1:])
        
        print(f"   Spatial correlation: {correlation:.4f}")
        print(f"   Variance: {pred_array.var():.6f}")
        
        if correlation > best_correlation:
            best_correlation = correlation
            best_factor = factor
            best_result = pred_array.copy()
        
        # Save this result with ORIGINAL SIZE
        normalized = (pred_array - pred_array.min()) / (pred_array.max() - pred_array.min() + 1e-8)
        normalized_uint8 = (normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and resize to original size
        colored_image = Image.fromarray(colored_rgb)
        colored_resized = colored_image.resize(original_size, Image.LANCZOS)
        
        output_path = os.path.join(output_dir, f"{base_name}_scale_{factor}.png")
        colored_resized.save(output_path)
        print(f"   Saved: {output_path} (size: {colored_resized.size})")
    
    print(f"\nBEST RESULT:")
    print(f"   Scaling factor: {best_factor}")
    print(f"   Spatial correlation: {best_correlation:.4f}")
    
    if best_correlation > 0.3:
        print(" Found structured output!")
        
        # Save best result with special name and ORIGINAL SIZE
        normalized = (best_result - best_result.min()) / (best_result.max() - best_result.min() + 1e-8)
        normalized_uint8 = (normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and resize to original size
        colored_image = Image.fromarray(colored_rgb)
        colored_resized = colored_image.resize(original_size, Image.LANCZOS)
        
        best_path = os.path.join(output_dir, f"{base_name}_BEST_scale_{best_factor}.png")
        colored_resized.save(best_path)
        print(f" Best result saved: {best_path} (size: {colored_resized.size})")
    else:
        print("  No scaling factor produced structured output")
    
    return best_result, best_factor

def debug_dataloader_format(image_path):
    """Debug what the actual DataLoader returns during training"""
    
    print(f"Debugging DataLoader format...")
    
    # Simulate what DataLoader_MS2 does
    from imageio import imread
    
    # 1. load_as_float_img output
    img_array = imread(image_path).astype(np.float32)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=2)
    
    print(f"load_as_float_img output:")
    print(f"   Shape: {img_array.shape}")
    print(f"   Range: [{img_array.min():.1f}, {img_array.max():.1f}]")
    print(f"   Dtype: {img_array.dtype}")
    
    # 2. What PyTorch DataLoader typically does
    # DataLoader often applies transforms automatically
    
    # Option A: Convert numpy to tensor (no normalization)
    tensor_no_norm = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    print(f"\nDirect numpy→tensor (no norm):")
    print(f"   Shape: {tensor_no_norm.shape}")
    print(f"   Range: [{tensor_no_norm.min():.1f}, {tensor_no_norm.max():.1f}]")
    
    # Option B: Apply ToTensor transform (normalizes to [0,1])
    from torchvision.transforms.functional import to_tensor
    if img_array.shape[-1] == 1:
        img_pil = Image.fromarray(img_array.squeeze().astype(np.uint8))
    else:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
    
    tensor_normalized = to_tensor(img_pil).unsqueeze(0)
    print(f"   Shape: {tensor_normalized.shape}")
    print(f"   Range: [{tensor_normalized.min():.4f}, {tensor_normalized.max():.4f}]")


if __name__ == "__main__":
    image_path = "F:/w1872042_FinalProjectCode/depth-estimation-model/000322_Night.png"
    model_path = "F:/w1872042_FinalProjectCode/depth-estimation-model/depth_estimation_sr3_250624_170128/checkpoints/full_model.pth"
    result1 = test_original_preprocessing(image_path, model_path)
    result2, best_factor = test_intermediate_preprocessing(image_path, model_path)
    debug_dataloader_format(image_path)