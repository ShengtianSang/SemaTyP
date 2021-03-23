<div align=center><img width="800" height="450" src="https://github.com/ShengtianSang/SemaTyP/blob/main/figures/Illustration_of_SemKG.jpg"/></div>

# SemaTyP: a knowledge graph based literature mining method for drug discovery

This is the source code and data for the task of drug discovery as describe in ou paper:
["SemaTyP: a knowledge graph based literature mining method for drug discovery"](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2167-5)

## Requirements
* scikit-learn
* numpy
* tqdm

## Data

In order to use your own data, you have to provide 
* [Theraputic Target Database]  You don't need to download by yourself, I have uploaded all the TTD 2016 version in *<./data/TTD>*. If you want a new version, click [here]((http://db.idrblab.net/ttd/full-data-download))
*
*

Process:
1) Input the command "pip install -r requirements.txt" to install the environment 

2) Using python command to construct training and test data: python experimental_data.py

3) Using python command to train and test the model: python main.py


>File declaration:

>><data> directory: contains all the data used in our experiments.
  
>>><data/SemmedDB> contains all relations extracted from SemmedDB, which are used for constructing the Knowledge Graph in our experiment. The whole "predications.txt" contains 39,133,975 relations, we just leave a small sample "predications.txt" file here which contain 100 relation. The whole "predications.txt" file coule be downloaded from 
  
>>><data/TTD> contains the drug, target and disease relations retrieved from Theraputic Target Database.
    
>><experimental_data>: constuct the drug-target-disease associations from TTD and Knowledge Graph.

>><knowledge_graph>: construct the Knowledge Graph used in our experiment.

>><models> directory: contains the trained models.
  
>><load_data.py> is used to load traing and test data.

>><main.py> is used to train and test the models

## Cite

Please cite our paper if you use this code in your own work:

```
@article{sang2018sematyp,
  title={SemaTyP: a knowledge graph based literature mining method for drug discovery},
  author={Sang, Shengtian and Yang, Zhihao and Wang, Lei and Liu, Xiaoxia and Lin, Hongfei and Wang, Jian},
  journal={BMC bioinformatics},
  volume={19},
  number={1},
  pages={1--11},
  year={2018},
  publisher={Springer}
}
```
