

# 🚀 Getting Started with RR-GCN (offical implementation of UAI2025 paper: Residual Reweighted Conformal Prediction for Graph Neural Networks)

## 1. Environment Setup 💻
First things first - let's get your tools ready! 

# Install with pip (recommended)
pip install torch torch_geometric

# For CUDA support (if you have NVIDIA GPU)
pip install torch torch_geometric --extra-index-url https://download.pytorch.org/whl/cu118

# Example of training and testing for Edge Weight Prediction is RR-GNN-GAE_DIGAE.ipynb and RR-GNN_LGNN.ipynb

# Example of training and testing  for Node Classification is RR-NodeClassficationF1.py


# Node Classification Example：

python RR-NodeClassficationF1.py


# Dataset Example:


# Traffic Network Datasets Specification

## 🚦 Anaheim Dataset

### 📊 Basic Information
- **Nodes**: 416 traffic intersections
- **Edges**: 914 road segments
- **Density**: ~1.06% (sparse network)
- **Average Node Degree**: 4.39

### 📍 Node Features (7-dimensional)
1. XY coordinates (latitude/longitude)
2. Traffic volume metrics:
   - AM peak hour traffic
   - PM peak hour traffic  
   - Daily total traffic
   - Truck percentage
   - Signal control indicator

### 🛣️ Edge Attributes (3-dimensional)
1. Travel time (minutes)
2. Road capacity (vehicles/hour)
3. Free-flow speed (mph)

### 🎯 Typical Applications
- Traffic flow prediction
- Congestion pattern analysis
- Transportation planning simulations


## 2. Chicago Transportation Dataset  
**Location**: Chicago, Illinois, USA  
**Use Cases**: Large-scale network analysis, transit optimization  

Key Statistics:
Nodes: 933 zones
Edges: 2,950 links
Density: ~0.68%
Avg Degree: 6.32
