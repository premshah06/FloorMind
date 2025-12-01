# Changelog

All notable changes to FloorMind will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-30

### Added
- Initial release of FloorMind
- Stable Diffusion XL integration for floor plan generation
- RESTful API with Flask backend
- React-based web interface
- GPU and CPU inference support
- Batch generation capabilities
- Health check and model info endpoints
- Preset prompts for common floor plan types
- Comprehensive error handling and logging
- Training scripts for custom models
- CubiCasa5K dataset processing pipeline
- Documentation (README, ARCHITECTURE, PROJECT_DOCUMENTATION)

### Features
- Generate 512×512 or 1024×1024 floor plans from text
- 5-10 second generation time on GPU
- 71.7% generation accuracy
- Fine-tuned on 5,050+ architectural floor plans
- DPM++ scheduler for fast, high-quality sampling
- Mixed precision (FP16) support for GPU
- Attention slicing for memory optimization
- Environment-based configuration
- CORS support for frontend integration

### Performance
- FID Score: 57.4
- CLIP Score: 0.75
- Model Size: ~5GB
- Supports NVIDIA GPUs with 8GB+ VRAM
- CPU fallback for systems without GPU

---

## Future Releases

### [1.1.0] - Planned
- [ ] Custom LoRA adapter support
- [ ] Improved prompt engineering
- [ ] Additional preset templates
- [ ] Performance optimizations
- [ ] Docker deployment support

### [1.2.0] - Planned
- [ ] Multi-floor building generation
- [ ] 3D floor plan export
- [ ] CAD software integration
- [ ] Advanced customization options

### [2.0.0] - Future
- [ ] Mobile app (iOS/Android)
- [ ] Cloud deployment templates
- [ ] Real-time collaboration features
- [ ] Advanced AI features

---

[1.0.0]: https://github.com/premshah06/FloorMind/releases/tag/v1.0.0
