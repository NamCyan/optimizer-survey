# Optimizer Survey

## Abstract
This study investigates optimization algorithms in Deep Learning, focusing on Gradient Descent. We explore different tasks, including Image and Text classification, as well as Image generation, employing 9 distinct optimization algorithms. The findings indicate that Adam-based techniques, such as QHAdam, AdamW, and Demon Adam, frequently outperform other methods in terms of learning effectiveness.

## Experiment
### Image classification
For this experiment, we utilize the CIFAR10 dataset. To assess both small and large model architectures, we employ LeNet and ResNet.

|     | LeNet                                                | ResNet                                             |
|     | Valid acc  | Valid F1    | Test acc     |Test F1     | Valid acc  | Valid F1    | Test acc   |Test F1     |
|:----|:-----------|------------:|-------------:|-----------:|-----------:|------------:|-----------:|-----------:|
|SGDM | 65.600     | 65.325      | 65.260       | 65.152     | 74.020     | 73.615      | 73.110     | 72.847     |
|Adam | 65.700     | 65.453      | 64.160       | 64.123     | 75.160     | 74.954      | 75.070     | 74.985     |
|AggMo| 65.560     | 65.031      | 65.490       | 65.109     | 72.760     | 72.367      | 71.990     | 71.641     |
|QHM  | **67.220** | **66.638**  | **65.860**   | **65.523** | 74.320     | 74.064      | 73.140     | 73.000     |
