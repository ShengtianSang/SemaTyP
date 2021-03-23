# SemaTyP: a knowledge graph based literature mining method for drug discovery
![SemKG]https://github.com/ShengtianSang/SemaTyP/blob/main/figures/Illustration_of_SemKG.jpg
This is the source code and data of our published paper "SemaTyP: a knowledge graph based literature mining method for drug discovery"

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
