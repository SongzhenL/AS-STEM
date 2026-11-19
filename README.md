<h2 align="center">
AS-STEM: Archimedean Spiral Enhanced Spatio-Temporal Embedding for Temporal Knowledge Graphs
</h2>



## 


### Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name teast_env python=3.8
```


### Datasets

```
python process_icews.py
python process_gdelt.py
```

This will create the files required to compute the filtered metrics.

### Reproducing results of AS-STEM

In order to reproduce the results of AS-STEM on the four datasets in the paper,  run the following commands

```
python  learner.py --dataset ICEWS14    --emb_reg 0.xx 

python  learner.py --dataset ICEWS05-15 --emb_reg xx

python  learner.py --dataset GDELT      --emb_reg xx
```
