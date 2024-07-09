# DNE-foundation-model-fairness

## Abstract
In the era of Foundation Models' (FMs) rising prominence in AI, our study addresses the challenge of biases in medical images while the model operates in black-box (e.g., using FM API), particularly spurious correlations between pixels and sensitive attributes. Traditional methods for bias mitigation face limitations due to the restricted access to web-hosted FMs and difficulties in addressing the underlying bias encoded within the FM API. We propose a D(ebiased) N(oise) E(diting) strategy, termed DNE, which generates DNE noise to mask such spurious correlation. DNE is capable of mitigating bias both within the FM API embedding and the images themselves. Furthermore, DNE is suitable for both white-box and black-box FM APIs, where we introduced G(reedy) (Z)eroth-O(rder) (GeZO) optimization for it when the gradient is inaccessible in black-box APIs. Our whole pipeline enables fairness-aware image editing that can be applied across various medical contexts without requiring direct model manipulation or significant computational resources. Our empirical results demonstrate the method's effectiveness in maintaining fairness and utility across different patient groups and diseases. In the era of AI-driven medicine, this work contributes to making healthcare diagnostics more equitable, showcasing a practical solution for bias mitigation in pre-trained image FMs.


## Schedule

- [x] Release the train dne code.
- [x] Release the finetune FM code.
- [ ] Release preprocessed datasets
- [ ] Release pretrained dne and models.
- [ ] Release the demo notebooks


## Mechanism
Modern FMs are not always accessible to user, e.g., those blackbox APIs. This means that the user can _only do linear probing_ using the FM's embedding during classificion, which prevents them from using traditional bias mitigation strategies. This motivate us to propose **DNE**. The mechanism of **DNE** is to train a vector (called **DNE**) that can be added on the image so that it can mask the sensitive-atrribute-related spurious correlation during training. **DNE** can be updated in both the white-box model and black-box model. 

## Demo
The following notebook provides a demo to use our pretrained DNE while finetuning the FM.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nanboy-Ronan/DNE-foundation-model-fairness/blob/main/finetune_fm_with_dne.ipynb)


## Pipeline
There are three steps to train the DNE to achieve the above performance.

### Step 1: Train a sensitive attribute classifier with FM
This step trains a sensitive-attribute classifier with FM using linear probing.

### Step 2 (white-box): Train the DNE using gradient descent if the model is accessible
If the FM is accessible, where user can access its gradient and architecture, then we can optimize DNE using the gradient descent.

### Step 2 (black-box): Train the DNE using GeZO (greey gradient zeroth-order optimization) 
If the FM is not accessible, e.g. through a black-box API. User can choose to apply GeZO to optimize the DNE where gradient is not needed.

### Step 3: Finetune the FM with DNE
Finally, we train the FM. Normally, user finetune the FM using linear probing, where they take the embedding of the FM and finetune its head. With DNE added on the input images, the finetune process become more fair.

## Contact
If you have any question, feel free to [email](mailto:ruinanjin@alumni.ubc.ca) us. We are happy to help you.
