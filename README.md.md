# GLEAM: Graph-Structured Latent Embedding with Adaptive Similarity Learning

Official implementation of **GLEAM** for Robust Multi-Label Classification. 

## 📖 Abstract
Multi-label classification in noisy, high-dimensional environments remains challenging due to redundant features, correlated labels, and local structural complexity. GLEAM is a unified framework that jointly optimizes feature projection, label projection, latent embedding, and an adaptive similarity graph. Unlike two-stage approaches, GLEAM employs a co-evolutionary mechanism where the adaptive graph is dynamically reconstructed from the projected latent space, imposing a data-dependent manifold constraint on the alignment process.

## 🛠️ Requirements
Install the required Python packages:
```bash
pip install -r requirements.txt