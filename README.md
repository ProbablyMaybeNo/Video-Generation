# Video-Generation
Local Video Generation Set Up by Cursor
# AI Video Generation Toolkit 🎬✨

A comprehensive, optimized environment for running state-of-the-art AI video generation models locally. Generate high-quality videos from images, videos, and text prompts using the latest AI models with both advanced and standard workflow options.

## 🚀 Features

### Advanced Video Generation Models
- **Stable Video Diffusion (SVD)** - Image-to-video generation
- **AnimateDiff** - Text-to-video with motion modules
- **I2VGen-XL** - High-resolution image-to-video
- **VideoCrafter** - Text and image-controlled video synthesis
- **CogVideoX** - Advanced text-to-video generation
- **Runway ML Gen-2** (API integration)
- **Pika Labs** (API integration)

### Standard/Quick Generation Models
- **Text2Video-Zero** - Fast text-to-video
- **ModelScope Text2Video** - Efficient baseline generation
- **Video-P2P** - Quick video editing and manipulation
- **FateZero** - Fast video editing with diffusion

### Input Formats Supported
- 📸 **Images**: JPG, PNG, WebP, TIFF
- 🎥 **Videos**: MP4, AVI, MOV, WebM
- 📝 **Text**: Natural language prompts, detailed descriptions
- 🎨 **Mixed**: Combine multiple input types

## 🛠️ Installation & Setup

### Prerequisites
```bash
# System Requirements
- NVIDIA GPU with 12GB+ VRAM (RTX 3090/4090 recommended)
- CUDA 11.8+ or 12.0+
- Python 3.9-3.11
- 32GB+ RAM recommended
- 100GB+ free storage
```

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-video-generation-toolkit.git
cd ai-video-generation-toolkit

# Create conda environment
conda create -n video-gen python=3.10
conda activate video-gen

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Setup Hugging Face access
huggingface-cli login

# Download base models
python setup_models.py --download-all
```

### Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

## 📁 Project Structure

```
ai-video-generation-toolkit/
├── models/                     # Model storage
│   ├── advanced/              # High-quality models
│   ├── standard/              # Fast models
│   └── custom/                # Fine-tuned models
├── workflows/                 # Generation pipelines
│   ├── text2video/           
│   ├── image2video/          
│   ├── video2video/          
│   └── mixed_input/          
├── scripts/                   # Utility scripts
├── config/                    # Configuration files
├── outputs/                   # Generated content
├── api/                       # REST API server
├── web_ui/                    # Gradio web interface
└── notebooks/                 # Jupyter examples
```

## 🎯 Quick Start

### 1. Text-to-Video Generation
```python
from video_gen import VideoGenerator

# Initialize with advanced model
generator = VideoGenerator(model="animatediff", quality="high")

# Generate video from text
video = generator.text_to_video(
    prompt="A serene lake at sunset with gentle ripples",
    duration=4.0,  # seconds
    fps=24,
    resolution=(1024, 576)
)

# Save output
video.save("outputs/lake_sunset.mp4")
```

### 2. Image-to-Video Generation
```python
# Initialize SVD model
generator = VideoGenerator(model="stable-video-diffusion")

# Generate from image
video = generator.image_to_video(
    image_path="inputs/landscape.jpg",
    motion_strength=0.7,
    duration=3.0
)
```

### 3. Quick Generation (Standard Models)
```python
# Fast generation for prototyping
quick_gen = VideoGenerator(model="text2video-zero", mode="fast")

video = quick_gen.text_to_video(
    prompt="Flying through clouds",
    duration=2.0,
    quality="standard"
)
```

## 🌐 Web Interface

Launch the interactive web interface:
```bash
python app.py --ui gradio
```

Access at `http://localhost:7860`

### Features:
- Drag-and-drop file uploads
- Real-time preview
- Batch processing
- Model comparison
- Parameter tuning
- Queue management

## 🔧 API Server

Start the REST API server:
```bash
python api_server.py --port 8000
```

### Example API Usage:
```bash
# Text-to-video
curl -X POST "http://localhost:8000/generate/text2video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing in a garden",
    "duration": 3.0,
    "model": "animatediff"
  }'

# Image-to-video
curl -X POST "http://localhost:8000/generate/image2video" \
  -F "image=@input.jpg" \
  -F "duration=4.0" \
  -F "model=stable-video-diffusion"
```

## ⚙️ Configuration

### Model Settings (`config/models.yaml`)
```yaml
advanced_models:
  stable_video_diffusion:
    model_path: "models/advanced/svd"
    vram_requirement: 12
    inference_time: "slow"
    quality: "high"
  
  animatediff:
    model_path: "models/advanced/animatediff"
    vram_requirement: 10
    inference_time: "medium"
    quality: "high"

standard_models:
  text2video_zero:
    model_path: "models/standard/t2v-zero"
    vram_requirement: 6
    inference_time: "fast"
    quality: "medium"
```

### Workflow Presets (`config/workflows.yaml`)
```yaml
presets:
  cinematic:
    aspect_ratio: "16:9"
    fps: 24
    duration: 5.0
    motion_strength: 0.8
    
  social_media:
    aspect_ratio: "9:16"
    fps: 30
    duration: 3.0
    motion_strength: 0.6
    
  preview:
    aspect_ratio: "16:9"
    fps: 12
    duration: 2.0
    motion_strength: 0.4
```

## 🚀 Advanced Workflows

### Batch Processing
```python
from video_gen.batch import BatchProcessor

processor = BatchProcessor()

# Process multiple prompts
prompts = [
    "Ocean waves crashing on rocks",
    "City skyline at night with moving traffic",
    "Forest with sunlight filtering through trees"
]

processor.batch_text_to_video(
    prompts=prompts,
    output_dir="outputs/batch/",
    model="animatediff"
)
```

### Custom Model Integration
```python
# Add your own fine-tuned model
from video_gen.models import register_custom_model

register_custom_model(
    name="my_custom_model",
    path="models/custom/my_model",
    config="config/custom_model.yaml"
)
```

### Pipeline Chaining
```python
# Chain multiple operations
pipeline = VideoGenerator.create_pipeline([
    ("upscale", {"factor": 2}),
    ("stabilize", {"strength": 0.5}),
    ("enhance", {"sharpness": 1.2})
])

enhanced_video = pipeline.process(original_video)
```

## 🔌 API Integrations

### Supported External APIs:
- **Runway ML Gen-2**: Premium quality generation
- **Pika Labs**: Advanced motion control
- **Stable Diffusion API**: Image generation
- **OpenAI DALL-E**: Image inputs
- **Claude/GPT**: Prompt enhancement

### Setup API Keys:
```bash
# Add to .env file
RUNWAY_API_KEY=your_runway_key
PIKA_API_KEY=your_pika_key
OPENAI_API_KEY=your_openai_key
```

## 📊 Performance Optimization

### GPU Memory Management
```python
# Automatic memory optimization
generator = VideoGenerator(
    model="stable-video-diffusion",
    memory_management="auto",
    enable_xformers=True,
    use_fp16=True
)
```

### Multi-GPU Support
```python
# Distribute across multiple GPUs
generator = VideoGenerator(
    model="animatediff",
    device_map="auto",
    gpu_ids=[0, 1, 2]
)
```

## 🔄 Autonomous Operation

### Scheduled Generation
```python
from video_gen.scheduler import VideoScheduler

scheduler = VideoScheduler()

# Schedule daily content generation
scheduler.add_daily_task(
    prompt_source="prompts/daily_prompts.txt",
    output_pattern="outputs/daily/{date}_{prompt_hash}.mp4",
    model="animatediff"
)

scheduler.start()
```

### Auto-Retry & Recovery
```python
# Automatic error handling and retries
generator = VideoGenerator(
    auto_retry=True,
    max_retries=3,
    fallback_model="text2video-zero"
)
```

## 📋 Requirements

### Python Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
xformers>=0.0.20
opencv-python>=4.8.0
pillow>=9.5.0
numpy>=1.24.0
gradio>=3.40.0
fastapi>=0.100.0
uvicorn>=0.23.0
```

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0

# For CUDA support
sudo apt install nvidia-cuda-toolkit
```

## 🚨 Troubleshooting

### Common Issues

**GPU Memory Error:**
```python
# Reduce batch size or use gradient checkpointing
generator = VideoGenerator(
    model="animatediff",
    gradient_checkpointing=True,
    batch_size=1
)
```

**Model Download Issues:**
```bash
# Manual model download
python scripts/download_models.py --model stable-video-diffusion --force
```

**CUDA Not Available:**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black video_gen/
isort video_gen/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stability AI for Stable Video Diffusion
- ByteDance for AnimateDiff
- DAMO Academy for I2VGen-XL
- Tencent for VideoCrafter
- All the open-source contributors

## 📞 Support

- **Documentation**: [Wiki](../../wiki)
- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Discord**: [Join our community](https://discord.gg/your-server)

---

**⭐ Star this repository if you find it helpful!**

Built with ❤️ for the AI video generation community
