# 2025 baby varnet
2025 SNU FastMRI challenge

## MRAugment

```bash
export PYTHONPATH=$PYTHONPATH:/root/FastMRI_challenge/MRAugment
```

## Step 1. Training classifier
```
terminal / train.sh / train.py / train_classifier.py / classifier.py / varnet.py
```

### Training hierarchy pseudocode
```
train.sh
  train.py
    train_classifier(args)
      model = VarNet(...) from varnet.py
        load VarNet_savefile
      classifier = AnatomyClassifier(...) from classifier.py
        train_epoch(...)
        validate(...)
        save Classifier_savefile
```

