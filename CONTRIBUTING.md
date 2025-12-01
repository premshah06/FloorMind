# Contributing to FloorMind

Thank you for your interest in contributing to FloorMind! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU/CPU)
- Error messages or logs

### Suggesting Features

Feature requests are welcome! Please open an issue describing:
- The feature you'd like to see
- Why it would be useful
- Possible implementation approach

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/premshah06/FloorMind.git
   cd FloorMind
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Test API
   python scripts/testing/test_api.py
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Ensure CI checks pass

## Development Setup

```bash
# Clone repository
git clone https://github.com/premshah06/FloorMind.git
cd FloorMind

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/
```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise

### Python Formatting

```bash
# Format code
black backend/ scripts/

# Check linting
flake8 backend/ scripts/

# Type checking
mypy backend/
```

## Testing

- Add tests for new features
- Ensure existing tests pass
- Test on both CPU and GPU if possible

## Documentation

- Update README.md if adding features
- Add docstrings to new functions
- Update ARCHITECTURE.md for architectural changes

## Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing! 🎉
