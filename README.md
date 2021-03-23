# SemaTyP: a knowledge graph based literature mining method for drug discovery

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


###########目录结构描述
├── Readme.md                   // help
├── app                         // 应用
├── config                      // 配置
│   ├── default.json
│   ├── dev.json                // 开发环境
│   ├── experiment.json         // 实验
│   ├── index.js                // 配置控制
│   ├── local.json              // 本地
│   ├── production.json         // 生产环境
│   └── test.json               // 测试环境
├── data
├── doc                         // 文档
├── environment
├── gulpfile.js
├── locales
├── logger-service.js           // 启动日志配置
├── node_modules
├── package.json
├── app-service.js              // 启动应用配置
├── static                      // web静态资源加载
│   └── initjson
│       └── config.js       // 提供给前端的配置
├── test
├── test-service.js
└── tools

