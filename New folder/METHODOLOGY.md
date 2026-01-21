# Methodology: Multi-Model Ensemble for Robust Image Classification

## 1. Model Architecture
Our system employs an ensemble of Deep Convolutional Neural Networks (CNNs) to leverage diverse inductive biases for improved generalization and robustness. The ensemble consists of two primary backbones:

1.  **ResNet-18** [1]: A residual network that utilizes skip connections to mitigate the vanishing gradient problem. We modify the fully connected layer to output $C=2$ class logits.
2.  **EfficientNet-B0** [2]: A scalable architecture optimized for efficiency and accuracy using compound scaling. We replace the final classifier layer for our binary task.

Optionally, a third model, **ConvNeXt-Tiny** [3], is integrated to introduce a modern, vision-transformer-inspired CNN architecture into the voting pool, further diversifying the feature representation.

## 2. Training Protocol
Each model $M_i$ is trained independently on the training set $\mathcal{D}_{train}$. We utilize the Adam optimizer with a learning rate of $1e-4$ and Cross-Entropy Loss:
$$ \mathcal{L}_{CE} = -\sum_{c=1}^C y_{o,c} \log(p_{o,c}) $$
where $y_{o,c}$ is the binary indicator (0 or 1) if class label $c$ is the correct classification for observation $o$, and $p_{o,c}$ is the predicted probability.

Images are resized to $224 \times 224$ and normalized using standard ImageNet mean and standard deviation: $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$.

## 3. Ensemble Strategy
We employ a **Soft Voting** mechanism for the final decision. For a given input $x$, each model outputs a probability distribution $P(y|x, M_i)$ via the Softmax function. The ensemble probability is calculated as the arithmetic mean of individual model probabilities:

$$ P_{ens}(y|x) = \frac{1}{N} \sum_{i=1}^{N} P(y|x, M_i) $$

Where $N$ is the number of models. This method captures the confidence of each model, distinguishing it from Hard Voting (majority rule).

### 3.1 Weighted Ensemble
To further refine predictions, we assign scalar weights $w_i$ to each model based on validation performance or confidence calibration:

$$ P_{weighted}(y|x) = \sum_{i=1}^{N} \hat{w}_i P(y|x, M_i) $$
where $\sum \hat{w}_i = 1$. This allows the system to prioritize stronger learners (e.g., EfficientNet) while still benefiting from the diversity of weaker learners.

## 4. Evaluation Metrics
We evaluate performance using:
*   **Accuracy**: The proportion of correct predictions.
*   **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve, measuring the model's ability to distinguish classes across different thresholds.

## References
[1] He, K., et al. "Deep residual learning for image recognition." CVPR 2016.
[2] Tan, M., & Le, Q. "EfficientNet: Rethinking model scaling for convolutional neural networks." ICML 2019.
[3] Liu, Z., et al. "A ConvNet for the 2020s." CVPR 2022.
