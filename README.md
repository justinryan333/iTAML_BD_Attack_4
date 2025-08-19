# iTAML - Backdoor Attack 
An implementation of "iTAML : An Incremental Task-Agnostic Meta-learning Approach". (CVPR 2020) [(paper link)](https://arxiv.org/abs/2003.11652). That has been modifed to perform a backdoor attack

Besides minor changes to the orginal codebase, the backdoor attack utilizes 2 extra files to perform the back door attack.
- data_bd.py: This file creates the poisoned dataset by utilizing a open-source (CIFAR10, CIFAR100, soon to be added: MNIST, Tiny-IMAGENET) and modifies specified images with a poison pattern. This is mainly used for the training part of the dataset but it also creates variation of the test data set to better understand the effects of the attack.
- cm.py / cm_2.py : both of these files create confusion matrix to better understand the results of the models accuracy scores. These are used after all the testing and training and use predictions from different locations of the meta model. cm.py creates a confusion matrix from the predictions of specifically the model during the meta-testing without knowledge. cm_2.py does the same however uses the predictions that come from the model during the meta-testing without knowledge.
