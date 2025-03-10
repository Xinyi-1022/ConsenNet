# ConsenNet
This is the official implementation of our IEEE TNNLS paper "Enhancing SSVEP-Based BCI Performance via Consensus Information Transfer Among Subjects".
## Abstract
The brain–computer interface (BCI) based on steady-state visual evoked potential (SSVEP) has received considerable attention for its high communication speed. While large datasets provide an important opportunity to enhance decoding accuracies, the key challenge lies in the exploration of existing data to extract valuable information based on the distinctive characteristics of brain responses. In this study, we introduce ConsenNet, a framework designed to enhance SSVEP classification performance by leveraging information from the diverse perspectives of existing subjects. First, this study exploits the diversity of existing subjects to generate new samples, which retain both task-related components and variability. This effectively enhances the network generalization capability on new subjects. Second, the structured knowledge that encapsulates the interrelationships between categories has been constructed and then transferred from the teacher network to the student network, guiding the student network to extract invariant features across subjects. Finally, our model incorporates a small amount of new subject data for model calibration in the final stage. Offline experiments conducted on three public datasets demonstrate the superiority of ConsenNet over 19 methods compared in this study, while online experiments validate its feasibility for real-world applications.

## Overview
![图片1](https://github.com/user-attachments/assets/ee37e023-653d-486f-a086-b23506d4d2ee)

ConsenNet employs a three-stage training scheme to enhance the decoding performance of a new subject by utilizing the EEG signal from the existing subjects. In Pretrain stage 1, the diversity of the existing subjects is used to augment training samples, and then they are converted to the frequency domain. The augmented EEGs in the frequency domain are used to train a powerful teacher network. The cross-entropy loss is used in this stage. In Pretrain stage 2, the parameters in the student network are initialized by inheriting the parameters from the teacher network, and the parameters in the teacher network are frozen. Then, the data from all existing subjects is averaged class by class. The averaged EEG can be viewed as the “Consensus EEG,” and it is fed into the teacher network to obtain the features of “Consensus EEG.” The contrastive loss transfers the structured knowledge from the teacher network to the student network. The total training loss in Pretrain stage 2 consists of the cross-entropy loss and the contrastive loss. In the fine-tuning stage, the student network is fine-tuned on a new subject by minimizing the cross-entropy loss. Then, the student network can be applied to this new subject for SSVEP decoding.

## Experimental settings
Each subject took turns being the new subject, and the other subjects (34 for the Benchmark dataset) were treated as existing subjects. For Pretrain stages 1 and 2, all the blocks from existing subjects were used as the training set. We randomly chose three subjects as the validation set. We provide the validation set we used for each testing subject in **Benchmark_validation_set.mat**. The model was optimized on the training set until the accuracy of the validation set stopped increasing in 100 epochs. For the fine-tuning stage, we used the first three blocks (calibration blocks) from the new subject as the training set and the rest of the blocks (test blocks) as the testing set.

## Pre-train stage 1
Pre-train the teacher network with the augmented dataset. Run **PreTrain_stage1.py**

## Pre-train stage 2
Pre-train the student network with the consensus data. Run **PreTrain_stage2.py**
To avoid redundant computations, we precomputed the average subject templates. The resulting tensor was precomputed and saved as **subj_templates.mat**. Its shape is [35, 40, 1250, 9].

## Fine-tuning stage
Finetune the student network with the subject-specific calibration data. Run **Finetune_stage.py**. 

# Citation
If you find our paper/code useful, please consider citing our work:
```
@article{zhang2024enhancing,
  title={Enhancing SSVEP-Based BCI Performance via Consensus Information Transfer Among Subjects},
  author={Zhang, Xinyi and Wei, Wei and Qiu, Shuang and Li, Xujin and Wang, Yijun and He, Huiguang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```
