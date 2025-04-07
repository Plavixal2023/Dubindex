# DubIndex - Global LLM Ranking System

A standardized system for evaluating and ranking Large Language Models using the DubIndex Formula.

## DubIndex Formula

```
DubIndex = 0.20×[Speed] + 0.30×[Hallucination Resistance] + 0.30×[Accuracy] + 
          0.10×[Efficiency] + 0.10×[Size Efficiency]
```

## Features

- Automatic LLM discovery and evaluation
- Global ranking API service
- Adaptive task routing
- Performance tracking and comparison
- Easy integration for developers

## Installation

```bash
pip install dubindex
```

## Usage

### As a Library

```python
from dubindex import AdaptiveLLMRouter

router = AdaptiveLLMRouter()
result = router.route_task("Write a Python function", {})
```

### As an API Service

```python
from dubindex.api import app as dubindex_api
import uvicorn

uvicorn.run(dubindex_api, host="0.0.0.0", port=8000)
```

## Documentation

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License
