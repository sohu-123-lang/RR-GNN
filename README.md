

# 🚀 Getting Started with RR-GCN

## 1. Environment Setup 💻
First things first - let's get your tools ready! 

# Install with pip (recommended)
pip install torch torch_geometric

# For CUDA support (if you have NVIDIA GPU)
pip install torch torch_geometric --extra-index-url https://download.pytorch.org/whl/cu118

#2 Example of training and testing for Edge Weight Prediction is RR-GNN-GAE_DIGAE.ipynb and RR-GNN_LGNN.ipynb

#3 Example of training and testing  for Node Classification is RR-NodeClassficationF1.py


#4 Node Classification Example：

python RR-NodeClassficationF1.py


#5 Dataset Example:


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

### Key Features  
- **Nodes**: Transit stations (1,000s)  
  - Attributes: Ridership, geographic coordinates  
- **Edges**: Transit routes (subway/bus lines)  
  - Weighted by: Frequency, passenger load  
- **Applications**:  
  - High-dimensional edge regression (e.g., `RR-GNN_LGNN.ipynb`)  
  - Critical node detection  

### Technical Highlights  
- Large-scale network tests model scalability  
- Multimodal features (temporal + geospatial)  
- Suitable for hybrid architectures (GNN+Transformer)  
=
