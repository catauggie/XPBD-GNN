# Hybrid XPBD-GNN and SpectralUNet for Composite Manufacturing Control

This repository implements a hybrid deep learning architecture for adaptive process control in fiber-reinforced composite manufacturing. It integrates graph-based physics modeling, 3D spectral segmentation, and an attention-driven control policy for real-time decision-making.

---

## 🔧 Model Overview

### 1. **XPBD-GNN**  
A physics-informed Graph Neural Network that simulates deformation and heat transfer in composite materials based on extended position-based dynamics (XPBD).

- **Input:**  
  - Node features (e.g., position, temperature)  
  - Edge features (e.g., spatial distance, conductivity)  
- **Output:**  
  - `F_pred`: Deformation gradient (4-dimensional)  
  - `theta_pred`: Scalar field (e.g., stress, temperature)  
- **Loss Function:**  
  - Combines thermal diffusion error and length constraint for physical consistency.

---

### 2. **SpectralUNet (3D Hyperspectral U-Net)**  
A 3D convolutional neural network that segments hyperspectral volumetric data to detect material defects or inhomogeneities.

- **Input:**  
  - Hyperspectral volume (e.g., shape `[B, C, H, W, D]`, where C is spectral bands)  
- **Output:**  
  - Voxel-wise class logits for defects, resin-rich zones, fiber misalignment, etc.  
- **Architecture:**  
  - Encoder-decoder with skip connections and 3D convolutions.

---

### 3. **CrossAttentionController**  
A lightweight controller that fuses learned graph-level features and hyperspectral summaries to output adaptive control signals.

- **Input:**  
  - Aggregated graph embedding from XPBD-GNN (`h_gnn`)  
  - Summary vector from HSI segmentation (`s_hsi`)  
- **Output:**  
  - Action vector (e.g., for process pressure, temperature, and resin feed rate)  
- **Mechanism:**  
  - Cross-attention between GNN and HSI feature spaces  
  - Normalized residual fusion  
  - Small MLP policy head

---

## 🧪 Example Pipeline

```python
# Step 1: Run GNN
F_pred, theta_pred, h = gnn(data)

# Step 2: Segment HSI volume
seg_logits = unet(hsi_tensor)

# Step 3: Fuse and compute actions
h_gnn = torch.mean(h, dim=0)  # Global graph feature
s_hsi = torch.tensor([0.8, 0.1, 0.1])  # Example defect summary vector
actions = ctrl(h_gnn, s_hsi)
