
# STGCN for Terrorism Risk Prediction

Torch implementation for **Spatial-Temporal Multi-Graph Convolutional Network-based Provincial-Day level Terrorism Risk Prediction**.

The entire project is divided into two folds: "Preprocessing_part" and "Prediction_part."  
## Prequirements
```
pip install -r requirements.txt
```
## Preprocessing_part: Data obtaining and multi-graph structures generation


Due to GTD restrictions (see https://www.start.umd.edu/gtd/terms-of-use/), the authors cannot open source the terrorist attack data here. 

If anyone wants to reproduce the dataset, please download the latest GTD "globalterrorismdb_0522dist.xlsx" (see https://www.start.umd.edu/gtd/contact/), and put it in the "Preprocessing_part/AFG_GTD_data/" fold. 

## Prediction_part: model training and prediction

Usage: run "main.py"(deep learning models) and "mainML.py" (classical machine learning models)
You can find pt file in save filefolder




