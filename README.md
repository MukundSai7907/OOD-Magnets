
![Screenshot_20221221_100103](https://user-images.githubusercontent.com/24581966/209053492-db014ab4-746a-4ad4-a5ae-3f956bc20c4b.png)
## Training
**(1) Stage 1: Supervised Contrastive Training**

Please refer [this repository](https://github.com/HobbitLong/SupContrast) for PyTorch implementation of Supervised Contrastive Learning [paper](https://arxiv.org/abs/2004.11362).


**(2) Stage 2: Virtual Outlier Synthesis**  

Following the [paper](https://arxiv.org/abs/2202.01197) and the [repository](https://github.com/deeplearning-wisc/vos), we generate virtual outliers. To synthesize a new set of virtual outliers, use `--generate_ood` while running the train script.

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
python stage4_ID_classifier.py --train
```


## Testing

**(1) OOD Classification**

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

**(2) ID Classification**
To use the model as a classifier for ID data:
```
python stage4_ID_classifier.py --test
```

## Visualization
 ```
python visualize_embeddings.py
```

Ensure we have the datasets for Textures (DTD), SVHN, iSUN, LSUN and Places 365 for testing and visualization


