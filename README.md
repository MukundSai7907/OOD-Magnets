# OOD-Detection
----

## Training
**(1) Stage 1: Supervised Contrastive Training**

Please refer [this repository](https://github.com/HobbitLong/SupContrast) for PyTorch implementation of Supervised Contrastive Learning [paper](https://arxiv.org/abs/2004.11362).


**(2) Stage 2: Virtual Outlier Synthesis**  

Following the paper and the repository, we generate virtual outliers. To synthesize a new set of virtual outliers, use `--generate_ood` while running the train script.

**(3) Stage 3: OOD Score Optimization**  

```
python train_virtual.py --epochs 20
  --learning_rate 0.1
  --batch_size 128
  --momentum 0.9
  --save 0.9 Folder_to_save_checkpoints
  --generate_ood # to generate a new set of virtual outliers
  --generate_centroids  # find ID class centroids

  
  
```
**(4) Stage 4: ID Classification Head Training**  

```
python train_ID_classifier.py
  
  
```


## Testing

To run the results for FPR95, AUROC and AUPR please run:

```
        python test.py --cifar_root <root path of the cifar dataset> (required)
                       --weights_path <path to weight of the model> (required)
                       --centroids_path <path to the centroids file> (required)
                       --batch_size <batch size of test batches>
                       --lsun_root <path to root directory of lsun dataset>
                       --places365_root <path to root directory of places dataset>
                       --isun_root <path to root directory of isun dataset>
                       --dtd_root <path to root directory of dtd dataset>
                       --svhn_root <path to root directory of svhn dataset>
```

For faster results it is better to run the script once per OOD dataset i.e with only one of the OOD datset path arguments at a time. 


## Visualization

