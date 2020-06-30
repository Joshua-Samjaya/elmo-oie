# Supervised OIE Pytorch
This repository provides a reimplementation of [supervised-oie](https://github.com/gabrielStanovsky/supervised-oie) with latest pytorch and pytorch-lightning. 

## Quickstart
1. Install requirements
```
pip install requirements.txt
```

2. Adjust hyperparemeters, e.g. `confidence.json`
```
cd ./hyperparams
vim confidence.json
```

3. Train model on GPU with GloVe
- download GloVe Embedding
- Edit GloVe path in <configuration_file>.json
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
python  train.py  --train=../data/train.oie.conll  --dev=../data/dev.oie.conll  --test=../data/test.oie.conll --load_hyperparams=<your-config.json> --gpus <num_gpus> --<other-flags>
```

## Important files or folders
```
./requirements.txt
./hyperparams
./src/model.py
./src/dataloader.py
./src/train.py
```

# Old README
Refer to [supervised-oie](https://github.com/gabrielStanovsky/supervised-oie)
