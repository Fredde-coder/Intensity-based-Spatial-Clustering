# Project Course in Data Science - Intensity-based Spatial Clustering
This repository is the experiments done for a paper written during studies at KTH. It implements two different clustering methods, [sFCM](https://www.sciencedirect.com/science/article/abs/pii/S0895611105000923) and a neural network clustering method, [DFC](https://ieeexplore.ieee.org/abstract/document/9151332), and runs them on synthetic brain images taken from the [brainweb project](https://github.com/casperdcl/brainweb).

## How to use
The default experiment is using the DFC.
```python run.py ```
To run the sFCM model use:
```python main.py  --model sFCM```


## Arguments
For a full list of arguments with explanations checkout the beginning of run.py



