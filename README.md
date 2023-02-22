# Repetition-exploration-SR
This is the source code for our reproducibility paper: "Repetition and Exploration in Sequential Recommendation: A Reproducibility Study"
# Main packages
- Recbole
- Pytorch


# Wandb

We use wandb to run, tune-hyperparameters and organize our results.

- Login to your wandb account, create a project and connect to it. See https://wandb.ai/quickstart for more details.
- Replace your project name in the following block in the train_main.py
```
wandb.init(project="YOUR PROJECT", 
            name=f"{config['method']}_{config['dataset']}_{config['loss_type']}_mode{config['train_mode']}_shared{config['shared_emb']}_fold{config['foldk']}",
            config=config)
```
- Method 1: using the following script to config and run:
```
train_main.py --dataset=diginetica --foldk=17 --method=sasrec --shared_emb=0 --train_mode=1 --hidden_size 64
```
- Method 2: using wandb sweep to run multiple versions on slurm:

#### An example yaml to create a sweep job in wandb

```
method: grid
metric:
  name: recall5
parameters:
  dataset:
    values:
      - diginetica
      - 16yoochoose
  foldk:
    values:
      - 7
      - 17
      - 27
      - 37
      - 57
  method:
    values:
      - gru4rec
      - sasrec
      - bert4rec
      - srgnn
      - repeatnet
      - caser
  hiden_size:
    values: [32, 64, 128]
  shared_emb:
    values:
      - 1
      - 0
  train_mode:
    values:
      - 0
      - 1
program: train_main.py
```
After you create the sweep task, you can start the sweep agent on slurm to run different versions in parallel, see https://docs.wandb.ai/guides/sweeps/start-sweep-agents for more details.

- The results will be shown in your wandb project automatically!! 

- Note that, you do not want to use wandb to run our code, just replace the wandb part in our code. It should be easy.
  
