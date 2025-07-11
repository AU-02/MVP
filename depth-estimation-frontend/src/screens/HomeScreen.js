import React, { useEffect, useState, useRef } from "react";

const HomeScreen = () => {
    const [user, setUser] = useState(null);
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");
    const [image, setImage] = useState(null);
    const [depthMap, setDepthMap] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [fileName, setFileName] = useState("");
    const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
    
    const imageRef = useRef(null);
    const depthRef = useRef(null);

    useEffect(() => {
        const token = localStorage.getItem("token");
        if (!token) {
            window.location.href = "/";
            return;
        }

        fetch("http://127.0.0.1:8000/home", {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Unauthorized or session expired");
            }
            return response.json();
        })
        .then(data => {
            setUser(data);
            setSuccess("Welcome back! Ready to generate depth maps.");
        })
        .catch(() => {
            setError("Session expired. Please log in again.");
            localStorage.removeItem("token");
            setTimeout(() => {
                window.location.href = "/";
            }, 2000);
        });
    }, []);

    // Auto-clear messages after 5 seconds
    useEffect(() => {
        if (error) {
            const timer = setTimeout(() => setError(""), 5000);
            return () => clearTimeout(timer);
        }
    }, [error]);

    useEffect(() => {
        if (success) {
            const timer = setTimeout(() => setSuccess(""), 5000);
            return () => clearTimeout(timer);
        }
    }, [success]);

    // Calculate optimal container dimensions based on image
    const getOptimalDimensions = (imgWidth, imgHeight) => {
        // For specific image size (900×247px), use exact dimensions
        if (imgWidth === 900 && imgHeight === 247) {
            return { width: 900, height: 247 };
        }
        
        // For other images, maintain aspect ratio but limit size
        const maxWidth = 900;
        const maxHeight = 300;
        const minWidth = 400;
        const minHeight = 150;
        
        // Use exact dimensions if within reasonable limits
        if (imgWidth <= maxWidth && imgHeight <= maxHeight && 
            imgWidth >= minWidth && imgHeight >= minHeight) {
            return { width: imgWidth, height: imgHeight };
        }
        
        // Scale to fit if too large
        const aspectRatio = imgWidth / imgHeight;
        let containerWidth = Math.min(maxWidth, imgWidth);
        let containerHeight = containerWidth / aspectRatio;
        
        if (containerHeight > maxHeight) {
            containerHeight = maxHeight;
            containerWidth = containerHeight * aspectRatio;
        }
        
        return { width: Math.round(containerWidth), height: Math.round(containerHeight) };
    };

    // Handle image load to get dimensions
    const handleImageLoad = (event, isOriginal = true) => {
        const img = event.target;
        const dimensions = getOptimalDimensions(img.naturalWidth, img.naturalHeight);
        
        if (isOriginal) {
            setImageDimensions(dimensions);
        }
    };

    // Handle Image Upload
    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            setError("Please select a valid image file");
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            setError("File size must be less than 10MB");
            return;
        }

        // Clear previous messages
        setError("");
        setSuccess("");
    
        // Display the uploaded image immediately
        setImage(URL.createObjectURL(file));
        setFileName(file.name);
        setDepthMap(null);
        setIsProcessing(true);
        setImageDimensions({ width: 0, height: 0 });
    
        const formData = new FormData();
        formData.append("file", file);
    
        const token = localStorage.getItem("token");
    
        fetch("http://127.0.0.1:8000/depth/depth-map", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
            },
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            setIsProcessing(false);
            if (data.depth_map_url) {
                setDepthMap(data.depth_map_url);
                setSuccess("Depth map generated successfully!");
            } else {
                throw new Error("No depth map URL received");
            }
        })
        .catch(error => {
            setIsProcessing(false);
            setError("Failed to generate depth map. Please try again with a different image.");
        });
    };

    // Handle download
    const handleDownload = async (imageUrl, filename) => {
        try {
            const response = await fetch(imageUrl);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            setSuccess("Image downloaded successfully!");
        } catch (error) {
            setError("Failed to download image");
        }
    };

    // Dynamic styles for image containers
    const getImageBoxStyle = (hasImage, hasDimensions = false) => ({
        width: hasDimensions && imageDimensions.width > 0 ? `${imageDimensions.width}px` : '900px',
        height: hasDimensions && imageDimensions.height > 0 ? `${imageDimensions.height}px` : '247px',
        border: hasImage ? '2px solid var(--accent)' : '2px dashed var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--card-bg)',
        borderRadius: '20px',
        overflow: 'hidden',
        transition: 'all 0.3s ease',
        position: 'relative',
        backdropFilter: 'blur(10px)',
        boxShadow: hasImage ? '0 0 30px var(--accent-light)' : 'var(--shadow)'
    });

    return (
        <div className="home-container">
            <div className="expanded-main-container">
                {/* LEFT SECTION - Upload Controls */}
                <div className="left-upload-section">
                    <h1 className="title">Diffusion Driven Depth Estimation</h1>
                    <p className="description">
                        Transform your NIR images into detailed depth maps using our diffusion model. 
                        Optimized for rectangular landscape images.
                    </p>
                    
                    {/* Success Message */}
                    {success && (
                        <div className="success-message">
                            {success}
                        </div>
                    )}
                    
                    {/* Error Message */}
                    {error && (
                        <div className="error-message">
                            {error}
                        </div>
                    )}
                    
                    {/* File Info */}
                    {fileName && (
                        <div className="file-info">
                            <strong>{fileName}</strong>
                            {imageDimensions.width > 0 && (
                                <div style={{ fontSize: '12px', marginTop: '5px', opacity: 0.8 }}>
                                    {imageDimensions.width} × {imageDimensions.height} pixels
                                </div>
                            )}
                        </div>
                    )}
                    
                    {/* Upload Button */}
                    <input 
                        id="file-upload" 
                        type="file" 
                        className="file-input" 
                        onChange={handleImageUpload}
                        accept="image/*"
                        disabled={isProcessing}
                    />
                    <label 
                        htmlFor="file-upload" 
                        className="upload-btn"
                        style={{ 
                            opacity: isProcessing ? 0.7 : 1,
                            cursor: isProcessing ? 'not-allowed' : 'pointer'
                        }}
                    >
                        {isProcessing ? (
                            <>🔄 Processing...</>
                        ) : (
                            <>📤 Upload NIR Image</>
                        )}
                    </label>

                    {/* Processing Info */}
                    {isProcessing && (
                        <div className="processing-info">
                            🧠 Diffusion model is analyzing your NIR image and generating depth map...
                        </div>
                    )}

                    {/* Tips Section */}
                    <div className="tips-section">
                        <strong>📋 Tips:</strong><br/>
                        • Use rectangular NIR images<br/>
                        • Best results with landscape format (900×247px ideal)<br/>
                        • Supported: JPG, PNG, GIF (max 10MB)<br/>
                    </div>
                </div>

                {/* RIGHT SECTION - Images Display */}
                <div className="right-images-section">
                    {/* Original Image */}
                    <div className="image-container">
                        <div className="image-label">📸 Original NIR Image</div>
                        <div 
                            className={`image-box ${image ? 'has-image' : ''}`}
                            style={getImageBoxStyle(image, true)}
                        >
                            {image ? (
                                <>
                                    <img 
                                        ref={imageRef}
                                        src={image} 
                                        alt="Uploaded NIR" 
                                        className="uploaded-image"
                                        onLoad={(e) => handleImageLoad(e, true)}
                                        style={{
                                            width: '100%',
                                            height: '100%',
                                            objectFit: 'contain',
                                            borderRadius: '18px',
                                            transition: 'all 0.3s ease',
                                            background: 'var(--card-bg)'
                                        }}
                                    />
                                    {/* Download button for original */}
                                    <button
                                        className="download-btn"
                                        onClick={() => handleDownload(image, `original_${fileName}`)}
                                        onMouseEnter={(e) => {
                                            e.target.style.opacity = 1;
                                            e.target.style.transform = 'scale(1.05)';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.target.style.opacity = 0.8;
                                            e.target.style.transform = 'scale(1)';
                                        }}
                                    >
                                        💾 Save Original
                                    </button>
                                </>
                            ) : (
                                <div className="placeholder-content">
                                    <div className="placeholder-icon">🖼️</div>
                                    <p className="placeholder-text">
                                        Upload a NIR image to see it here
                                    </p>
                                    <p className="placeholder-subtext">
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Depth Map */}
                    <div className="image-container">
                        <div className="image-label">🎯 Generated Depth Map</div>
                        <div 
                            className={`image-box ${depthMap ? 'has-image' : ''}`}
                            style={getImageBoxStyle(depthMap, depthMap && imageDimensions.width > 0)}
                        >
                            {isProcessing ? (
                                <div className="loading-content">
                                    <div className="loading-icon">🔄</div>
                                    <p className="loading-text">Generating depth map...</p>
                                    <div className="loading-bar">
                                        <div className="loading-progress"></div>
                                    </div>
                                    <p className="loading-subtext">
                                        Processing your NIR image...
                                    </p>
                                </div>
                            ) : depthMap ? (
                                <>
                                    <img 
                                        ref={depthRef}
                                        src={depthMap} 
                                        alt="Depth Map" 
                                        className="depth-image"
                                        onLoad={(e) => handleImageLoad(e, false)}
                                        style={{
                                            width: '100%',
                                            height: '100%',
                                            objectFit: 'contain',
                                            borderRadius: '18px',
                                            transition: 'all 0.3s ease',
                                            background: 'var(--card-bg)'
                                        }}
                                    />
                                    {/* Download button for depth map */}
                                    <button
                                        className="download-btn"
                                        onClick={() => handleDownload(depthMap, `depth_map_${fileName}`)}
                                        onMouseEnter={(e) => {
                                            e.target.style.opacity = 1;
                                            e.target.style.transform = 'scale(1.05)';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.target.style.opacity = 0.8;
                                            e.target.style.transform = 'scale(1)';
                                        }}
                                    >
                                        💾 Save Depth Map
                                    </button>
                                </>
                            ) : (
                                <div className="placeholder-content">
                                    <div className="placeholder-icon">🎯</div>
                                    <p className="placeholder-text">
                                        Depth map will appear here
                                    </p>
                                    <p className="placeholder-subtext">
                                        Upload an image to generate its depth map
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HomeScreen;