# 2025 baby varnet
2025 SNU FastMRI challenge

## MRAugment

```bash
export PYTHONPATH=$PYTHONPATH:/root/FastMRI_challenge/MRAugment
```

## Step 1. Training classifier
```
terminal / train.sh / train.py / train_classifier.py / varnet.py / classifier.py
```

**Training hierarchy pseudocode**
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

## Step 2. Training teacher
```
terminal / train.sh / train.py / train_teacher.py / teacher_varnet.py / unet_for_distill.py
```

**Training hierarchy pseudocode**
```
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
terminal / train.sh / train.py / train_student.py / student_varnet.py / unet_for_distill.py
```

**Training hierarchy pseudocode**
```
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

## Step 4. Reconstruction
```
terminal / reconstruct.sh / reconstruct.py / test_part.py
```

**Training hierarchy pseudocode**
```
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

# Main model training
## Trainging teacher