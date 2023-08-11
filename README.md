# Optimizer Survey

## Abstract
This study investigates optimization algorithms in Deep Learning, focusing on Gradient Descent. We explore different tasks, including Image and Text classification, as well as Image generation, employing 9 distinct optimization algorithms. The findings indicate that Adam-based techniques, such as QHAdam, AdamW, and Demon Adam, frequently outperform other methods in terms of learning effectiveness.

## Experiment
### Image classification
For this experiment, we utilize the CIFAR10 dataset. To assess both small and large model architectures, we employ LeNet and ResNet.

| LeNet/ResNet | Valid acc  | Valid F1    | Test acc     |Test F1     | Valid acc  | Valid F1    | Test acc   |Test F1     |
|:-------------|:-----------|------------:|-------------:|-----------:|-----------:|------------:|-----------:|-----------:|
|SGDM          | 65.600     | 65.325      | 65.260       | 65.152     | 74.020     | 73.615      | 73.110     | 72.847     |
|Adam          | 65.700     | 65.453      | 64.160       | 64.123     | 75.160     | 74.954      | 75.070     | 74.985     |
|AggMo         | 65.560     | 65.031      | 65.490       | 65.109     | 72.760     | 72.367      | 71.990     | 71.641     |
|QHM           | **67.220** | **66.638**  | **65.860**   | **65.523** | 74.320     | 74.064      | 73.140     | 73.000     |
|DemonSGD      | 65.440     | 65.350      | 65.070       | 65.118     | 73.100     | 72.627      | 73.280     | 73.118     |
|AMSGrad       | 65.020     | 64.684      | 64.700       | 64.700     | 75.120     | 74.793      | 73.660     | 72.512     |
|QHAdam        | 65.340     | 64.953      | 64.600       | 64.554     | 75.560     | 75.672      | **75.130** | **75.428** |
|DemonAdam     | 65.280     | 64.847      | 65.270       | 64.995     | 75.260     | 75.106      | 74.200     | 74.235     |
|AdamW         | 64.540     | 64.081      | 63.110       | 62.755     | **76.000** | **75.885**  | 75.030     | 75.081     |

