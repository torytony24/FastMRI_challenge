# Universal VarNet
2025 SNU FastMRI challenge

```bash
pip3 install -r requirements.txt
apt-get update -y && apt-get install -y libgl1-mesa-glx
```

## MRAugment

```bash
export PYTHONPATH=$PYTHONPATH:/root/FastMRI_challenge/MRAugment
```

## Step 1. Training classifier
```bash
sh train_classifier.sh
```

**Training hierarchy pseudocode**
```python
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

## Step 2. Training teacher
```
sh train_teacher.sh
```

**Training hierarchy pseudocode**
```python
train.sh
  train.py
    train_teacher(args)
      model = Teacher_VarNet(...) from teacher_varnet.py
        train_identifier = {brain: 0, knee: 1}
        train_epoch(...)
        validate(...)
        save Teacher_brain_savefile & Teacher_knee_savefile
```

## Step 3. Training student
```
sh train_student.sh
```

**Training hierarchy pseudocode**
```python
train.sh
  train.py
    train_student(args)
      model_teacher = Teacher_VarNet(...) from teacher_varnet.py
        load Teacher_brain_savefile & Teacher_knee_savefile
        get feature_teacher
      model_student = Student_VarNet(...) from student_varnet.py
        distill feature_teacher to feature_student
        train_epoch(...)
        validate(...)
```

## Step 4. Reconstruction & Evaluation
```
sh reconstruct.sh
sh leaderboard_eval.sh
```

**Training hierarchy pseudocode**
```python
reconstruct.sh
  reconstruct.py
    forward(args)
      model_cls = VarNet(...) from varnet.py
        load VarNet_savefile
      classifier = AnatomyClassifier(...) from classifier.py
        load Classifier_savefile
      model_student = Student_VarNet(...) from student_varnet.py
        feature = model_cls(input)
        pred = classifier(feature)
        output = model_student(input, pred)
```








