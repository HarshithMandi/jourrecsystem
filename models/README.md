# Models Directory

This directory contains the pre-downloaded BERT model for offline use.

## Contents

### all-MiniLM-L6-v2
- **Type**: Sentence Transformer (MiniLM variant)
- **Dimensions**: 384
- **Size**: ~80MB
- **Purpose**: General-purpose semantic similarity for journal recommendation
- **Source**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Offline Usage

The application automatically uses the local model from this directory, eliminating the need for internet connectivity after initial setup.

### How It Works

1. When the application starts, it checks if `models/all-MiniLM-L6-v2/` exists
2. If found, it loads the model from this local directory
3. If not found, it downloads the model from HuggingFace (requires internet)
4. The downloaded model is saved here for future offline use

## Benefits

✓ **No internet required** - Works completely offline after setup
✓ **Faster startup** - No download checks or network delays  
✓ **Consistent versions** - Specific model version is locked
✓ **Portable** - Can be deployed to air-gapped environments

## Regenerating the Model

If you need to re-download the model:

```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); m.save('models/all-MiniLM-L6-v2')"
```

## Version Control

The model files (~80MB) are included in the repository for offline deployment.
If you prefer not to commit them to git, add to `.gitignore`:

```
models/all-MiniLM-L6-v2/
```

Then distribute the model separately or let users download on first run.
