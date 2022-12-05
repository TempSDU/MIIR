# Modeling Sequential Recommendation as Missing Information Imputation

> Codes for the submission: Modeling Sequential Recommendation as Missing Information Imputation

## Introduction

Side information is being used extensively to improve the effectiveness of sequential recommendation models.  It is said to help capture the transition patterns among items. Most previous work on sequential recommendation that uses side information models item IDs and side information separately. This fails to encode relations between items and their side information. Moreover, in real-world systems, not all values of item feature fields are available. This hurts the performance of models that rely on side information. Existing methods tend to neglect the context of missing item feature fields, and fill them with generic or special values, e.g., *unknown*, which might lead to sub-optimal performance.

To address the limitation of sequential recommenders with side information, we define a way to fuse side information and alleviate the problem of missing side information by proposing a unified task, namely the *missing information imputation* (MII), which randomly masks some feature fields in a given sequence of items, including item IDs, and then forces a predictive model to recover them. By considering the next item as a missing feature field, sequential recommendation can be formulated as a special case of MII. We propose a sequential recommendation model, called *missing information imputation recommender* (MIIR), that builds on the idea of MII and simultaneously imputes missing item feature values and predicts the next item. We devise a *dense fusion self-attention* (DFSA) for MIIR to directly capture relations between items and their side information.

## Requirements

To install the required packages, please run:

```python
pip install -r requirements.txt
```

Note that we use *accelerate* to configure the training environment, please refer to the [official website](https://huggingface.co/docs/accelerate/index) to revise *accelerate_config.yaml*.

## Datasets

We use *Beauty*, *Sports and Outdoors* and *Toys and Games* datasets for experiments. Please download them from the [Amazon review data](https://nijianmo.github.io/amazon/index.html) and put them in folders `/data/bt/raw`,  `/data/so/raw` and `/data/tg/raw` respectively. Then run the following commands to process the datasets:

```python
python ProcessBT.py
python ProcessSO.py
python ProcessTG.py
```

In order to easily switch the datasets in experiments, we use *bt_files.dat* and *tg_files.dat* to record all filepaths. The experiments are performed on the datasets after discard by default. For running the experiments on the datasets before discard, please revise *bt_files.dat*, *so_files.dat* and *tg_files.dat*:

```json
'masks_filepath': None
```

### Experiments

For training the MIIR by the MII loss, validating and tesing on the *Beauty* dataset, please run:

```python
accelerate launch --config_file accelerate_config.yaml Pretrain.py -dataset_files bt_files.dat -save_path checkpoint_miirs_bt/
accelerate launch --config_file accelerate_config.yaml Prevalid.py -dataset_files bt_files.dat -save_path checkpoint_miirs_bt/
accelerate launch --config_file accelerate_config.yaml Pretest.py -dataset_files bt_files.dat -save_path checkpoint_miirs_bt/
```

For training/fintuning the MIIR by the recommendation loss, validating and tesing on the *Beauty* dataset, please run:

```python
accelerate launch --config_file accelerate_config.yaml Train.py -dataset_files bt_files.dat -save_path checkpoint_miirs_bt/  # if need to finetune the MIIR pretrained by the MII loss, please use -pretrained_model 
accelerate launch --config_file accelerate_config.yaml Valid.py -dataset_files bt_files.dat -save_path checkpoint_miirs_bt/
accelerate launch --config_file accelerate_config.yaml Test.py -dataset_files bt_files.dat -save_path checkpoint_miirs_bt/
```

For the *Sports and Outdoors* (*Toys and Games*) dataset, please switch *bt_files.dat* to *so_files.dat* (*tg_files.dat*) and `checkpoint_miirs_bt/` to `checkpoint_miirs_so/` (`checkpoint_miirs_tg/`) in the above commands.

If you want to run the MIIR-M which is the variant masking missing feature fields in self-attention, please replace the original TrainDataset_MII, TrainDataset_MLM and TestDataset_MLM with the annotated ones in *DataLoader.py*. And delete the following codes in *MIIR.py*:

```python
padding_mask = padding_mask.unsqueeze(2)  # [batch_size, seq_len, 1]
padding_mask = padding_mask.repeat(1, 1, len(self.feature_fields)+1)  # [batch_size, seq_len, field_num]
```

If you want to run MIIR with the sparse fusion self-attention (SFSA), please free the following codes in *MIIR.py*:

```python
cross_mask = self.generate_cross_mask(shape[1], shape[2])
temps = mod(temps, src_key_padding_mask=padding_mask, src_mask=cross_mask)  # and delete temps = mod(temps, src_key_padding_mask=padding_mask) 
```
