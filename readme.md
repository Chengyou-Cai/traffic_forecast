# Traffic Forecast

## Dataset

|    |Time Steps|Nodes|Edges|Description|
|----|----|----|----|----|
|METR-LA|34272|207|1515|洛杉矶高速公路207个检测器2012年4个月的数据|
|PEMS-BAY|52116|325|2369|加州交通测量系统325个检测器2017年6个月的数据|

Download Links：[Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) & [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g#list/path=%2F) Provided By [DCRNN](https://github.com/liyaguang/DCRNN)

## Models

### Graph WaveNet

#### Reference

* [Graph WaveNet for Deep Spatial-Temporal Graph Modeling](https://arxiv.org/abs/1906.00121)
* [Incrementally Improving Graph WaveNet Performance on Traffic Prediction](https://arxiv.org/abs/1912.07390)

#### Simple Comparison Results on METR-LA

|Method|60min MAE (Vehs)|
|----|----|
|DCRNN|3.60|
|[GWN](https://github.com/nnzhan/Graph-WaveNet)|3.55|
|[GWNV2](https://github.com/sshleifer/Graph-WaveNet) (models/gwavenet.py)|3.45|
|GWN_M (models/gwavenet_mine.py)|3.32|

### Traffic Transformer

#### Reference

* [Traffic transformer: Capturing the continuity and periodicity of time series for traffic forecasting](https://onlinelibrary.wiley.com/doi/abs/10.1111/tgis.12644)

## Visualization App

streamlit is being used to build the app.

## Pipeline of Commands
