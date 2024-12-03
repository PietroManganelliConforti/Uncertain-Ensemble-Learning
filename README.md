# Uncertain-Ensemble-Learning

### HOW TO RUN:

To clone the repo:
```
git clone https://github.com/PietroManganelliConforti/Uncertain-Ensemble-Learning.git
```
To train the models:
```
docker run -v $PWD/:/work/project/ -it  --gpus all --ipc host piemmec/adv_unc  python3 work/project/main.py
```
To launch the FGSM attack:
```
docker run -v $PWD/:/work/project/ -it  --gpus all --ipc host piemmec/adv_unc  python3 work/project/fgsm.py
```
To launch the training script from docker, in detached mode:
```
docker run -v $PWD/:/work/project/ -d --gpus all --ipc host piemmec/adv_unc sh work/project/script.sh
```

### Useful links:

Google doc: https://docs.google.com/document/d/1eTaxMID_BfD9CypqR-XnWq72kII3ESS2fnmFBboxRY4/edit?tab=t.0

Overleaf: https://www.overleaf.com/project/673dadd5e7aac9f5cdacc679

Slack: https://app.slack.com/client/T07UKKVN004/C07TBJAGSUA

