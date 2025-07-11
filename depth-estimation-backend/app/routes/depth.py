from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from scipy.stats import pearsonr
from app.core.security import get_current_user  

# Path to model directory
sys.path.insert(0, "F:/w1872042_FinalProjectCode/data")
sys.path.insert(0, "F:/w1872042_FinalProjectCode/depth-estimation-model/depth_estimation_sr3_250624_170128")

router = APIRouter()

# Directory to save uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model configuration
MODEL_CHECKPOINT = "F:/w1872042_FinalProjectCode/depth-estimation-model/depth_estimation_sr3_250624_170128/checkpoints/full_model.pth"

def test_original_preprocessing(image_path, output_dir=UPLOAD_DIR):
    """Test with the original [0,1] preprocessing"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=False)
    model.eval()
    
    # ORIGINAL preprocessing
    image = Image.open(image_path).convert('L').resize((256, 256))
    
    # Original approach
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
    
    return pred_array, output_path, correlation

def test_intermediate_preprocessing(image_path, output_dir=UPLOAD_DIR):
    """Test with intermediate normalization values"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=False)
    model.eval()
    
    # Try different scaling factors
    scaling_factors = [1.0, 2.0]  # Different ways to scale [0,1] back up
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Get original size
    original_size = Image.open(image_path).size
    print(f"Original image size: {original_size}")
    
    best_correlation = 0
    best_factor = None
    best_result = None
    best_output_path = None
    all_results = []
    
    for factor in scaling_factors:
        print(f"\nTesting scaling factor: {factor}")
        
        try:
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
            
            # Store result info for API response
            all_results.append({
                "scaling_factor": factor,
                "spatial_correlation": round(correlation, 4),
                "variance": round(pred_array.var(), 6),
                "min_value": round(pred_array.min(), 4),
                "max_value": round(pred_array.max(), 4),
                "structured": correlation > 0.3,
                "output_file": f"{base_name}_scale_{factor}.png"
            })
            
        except Exception as e:
            print(f"   Error with factor {factor}: {e}")
            all_results.append({
                "scaling_factor": factor,
                "error": str(e)
            })
    
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
        
        best_output_path = os.path.join(output_dir, f"{base_name}_BEST_scale_{best_factor}.png")
        colored_resized.save(best_output_path)
        print(f" Best result saved: {best_output_path} (size: {colored_resized.size})")
    else:
        print("  No scaling factor produced structured output")
    
    return best_result, best_factor, best_correlation, best_output_path, all_results

def debug_dataloader_format(image_path):
    """Debug what the actual DataLoader returns during training"""
    
    print(f"Debugging DataLoader format...")
    
    try:
        # Simulate what DataLoader_MS2 does
        from imageio import imread
        
        # 1. load_as_float_img output
        img_array = imread(image_path).astype(np.float32)
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
    except ImportError:
        print("imageio not available, using PIL instead")
        img_array = np.array(Image.open(image_path)).astype(np.float32)
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
    print(f"ToTensor transform (normalized):")
    print(f"   Shape: {tensor_normalized.shape}")
    print(f"   Range: [{tensor_normalized.min():.4f}, {tensor_normalized.max():.4f}]")

@router.post("/depth-map")
async def generate_depth_map(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Generate and return only the best depth map image URL"""

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Run preprocessing tests
        print("Running original preprocessing test...")
        original_result, original_path, original_correlation = test_original_preprocessing(file_path)

        print("\nRunning intermediate preprocessing tests...")
        best_result, best_factor, best_correlation, best_output_path, all_results = test_intermediate_preprocessing(file_path)

        # Choose best result
        if best_output_path and best_correlation > original_correlation:
            depth_filename = os.path.basename(best_output_path)
        else:
            depth_filename = os.path.basename(original_path)

        return {
            "depth_map_url": f"http://127.0.0.1:8000/depth/uploads/{depth_filename}"
        }

    except Exception as e:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/depth-map-detailed")
async def generate_depth_map_detailed(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Generate depth map with detailed analysis"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Run ALL tests
        print("Running original preprocessing test...")
        original_result, original_path, original_correlation = test_original_preprocessing(file_path)
        
        print("\nRunning intermediate preprocessing tests...")
        best_result, best_factor, best_correlation, best_output_path, all_results = test_intermediate_preprocessing(file_path)
        
        print("\nRunning dataloader format debug...")
        debug_dataloader_format(file_path)
        
        # Prepare response with all files
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        response_data = {
            "message": "Complete depth map analysis finished - EXACT working script replication",
            "original_preprocessing": {
                "spatial_correlation": round(original_correlation, 4),
                "file_url": f"http://127.0.0.1:8000/depth/uploads/{base_name}_original_preprocessing.png"
            },
            "best_result": {
                "scaling_factor": best_factor,
                "spatial_correlation": round(best_correlation, 4) if best_correlation else 0,
                "file_url": f"http://127.0.0.1:8000/depth/uploads/{os.path.basename(best_output_path)}" if best_output_path else None
            },
            "all_scaling_results": all_results,
            "scale_files": [f"http://127.0.0.1:8000/depth/uploads/{result.get('output_file', '')}" 
                          for result in all_results if 'output_file' in result],
            "method": "exact_working_script_replication"
        }
        
        return response_data
        
    except Exception as e:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Detailed analysis failed: {str(e)}")

@router.post("/debug-dataloader")
async def debug_dataloader_endpoint(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Debug dataloader format"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Run debug function
        debug_dataloader_format(file_path)
        
        return {"message": "Dataloader format debug completed - check server logs for details"}
        
    except Exception as e:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

# Endpoint to serve uploaded files
@router.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """Serve uploaded files and generated depth maps"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

# Health check endpoint
@router.get("/health")
async def health_check():
    """Check if the depth estimation service is working"""
    try:
        return {
            "status": "healthy", 
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "method": "exact_working_script_replication"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__
        }