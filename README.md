# Supervised OIE Pytorch
This repository provides a reimplementation of [supervised-oie](https://github.com/gabrielStanovsky/supervised-oie) with latest pytorch and pytorch-lightning. 

## Quickstart
1. Install requirements
```
pip install requirements.txt
```

2. Create configuration file with hyperparemeters, e.g. `confidence.json`
```
cd ./hyperparams
vim confidence.json
```

3. Train model on GPU with GloVe
- Download GloVe Embedding
- Edit GloVe path in <confign_file>.json
```
cd ./src
python  train.py  --train=../data/train.oie.conll  --dev=../data/dev.oie.conll  --test=../data/test.oie.conll --load_hyperparams=../hyperparams/confidence.json --gpus 1
```

4. Train model on GPU with BERT
```
cd ./src
python  train.py  --train=../data/train.oie.conll  --dev=../data/dev.oie.conll  --test=../data/test.oie.conll --load_hyperparams=../hyperparams/confidence.json --gpus 1 --use-bert
```

5. Train model with customized settings
- Edit hyperparameter json file. 
- Run the following code
```
cd ./src
python  train.py  --train=../data/train.oie.conll  --dev=../data/dev.oie.conll  --test=../data/test.oie.conll --load_hyperparams=<config_file.json> --gpus <num_gpus> --<other-flags>
```

## Important files or folders
```
./requirements.txt
./hyperparams
./src/model.py
./src/dataloader.py
./src/train.py
```

## To contribute
1. Please try to understand the important files or folders. 
2. Please create another branch first, say "ELMo". Then edit and commit with your code in that branch. A sample "elmo" branch has been created, you may do:
```
git checkout elmo
git status
```
3. Please create your own json file under `./hyperparams`. 
4. Please create your own dataloader and model class under `./src`. 
5. Please try to separate your code with the existing as much as possible. 

# Old README
Refer to [supervised-oie](https://github.com/gabrielStanovsky/supervised-oie)
