# AnomalyDetection

## Software dependencies 

1. The source code is written in Python with Tensorflow (including Keras) and Jupyter. Please make sure you have these tools installed properly in your system.
   - The version of Python used: 3.6.9
   - The version of Tensorflow used: 1.12.0
   - The version of  Keras used: 2.2.4
   - The version of the notebook server is: 6.0.1

## Datasets

 The  OOD datasets can be downloaded automatically by opening *"download_required_oods.ipynb"* and executing all cells. If this process goes well, there will be two folders named *"ood_datasets"* and  beside the file *"download_required_files.ipynb"*. 

## Models

We pre-trained five neural networks (1) LeNet on MNIST;  (2) two VGG and ResNet models trained on Cifar10 and SVHN, respectively. 

You may download pre-trained models from 
https://1drv.ms/u/s!AmxoS1DPJxUZhyV8OAIvz6Yqji_w?e=TzoGWN

Please place them to './saved_models/'.

## Detecting Out-of-Distribution Samples

 To experiment with our approaches and others listed in the paper, please open *"Performance_evaluation-OOD.ipynb"* and execute cells to get the results based on the metrics defined in the paper. If you have your own model and want to find the OODL and repeat the experiments, you first need to use *"find_oodl.ipynb"* for find the OODL for your model. 

- By *id_name*, you can set the ID dataset. The list of ID datasets is: ("MNIST", "CIFAR10", "SVHN")

- By *id_model*, you can set the ID model. The list of ID models is: ("LeNet", "VGG", "ResNet")

- By *ood_appr_name*, you can select the OOD detection approach. The list of OOD detection approaches is: ("ours", "ours_w_p" , "softmax", "odin_wo_p", "odin_w_p", "mah_dist_logits", "mah_dist_logits_w_p")

Please place the ID and OOD sub-classes datas  to './classdatas_id/' and './classdatas_ood/'   before  executing the above cells. 

## Detecting Adversarial Samples

 To experiment with our approaches and others listed in the paper, please open *"Performance_evaluation-adversarial.ipynb"* and execute cells to get the results based on the metrics defined in the paper. If you have your own model and want to find the OODL and repeat the experiments, you first need to use *"find_oodl.ipynb"* for find the OODL for your model. 

- By *id_name*, you can set the ID dataset. The list of ID datasets is: ("MNIST", "CIFAR10", "SVHN")

- By *id_model*, you can set the ID model. The list of ID models is: ("LeNet", "VGG", "ResNet")

- By *ood_appr_name*, you can select the OOD detection approach. The list of OOD detection approaches is: ("ours", "ours_w_p" ,  "mah_dist_logits", "mah_dist_logits_w_p")
- You may download codes of KD+BU and LID  from https://github.com/xingjunm/lid_adversarial_subspace_detection

Please place the ID, adversarial  sample and adversarial  sub-classes datas  to './classdatas_id/',  './adv_datas/' and './classdatas_adv/'   before  executing the above cells. 

