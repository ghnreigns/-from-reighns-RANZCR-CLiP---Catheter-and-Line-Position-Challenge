---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Introduction

> Extracted from Kaggle 

Serious complications can occur as a result of malpositioned lines and tubes in patients. Doctors and nurses frequently use checklists for placement of lifesaving equipment to ensure they follow protocol in managing patients. Yet, these steps can be time consuming and are still prone to human error, especially in stressful situations when hospitals are at capacity.

Hospital patients can have catheters and lines inserted during the course of their admission and serious complications can arise if they are positioned incorrectly. Nasogastric tube malpositioning into the airways has been reported in up to 3% of cases, with up to 40% of these cases demonstrating complications [1-3]. Airway tube malposition in adult patients intubated outside the operating room is seen in up to 25% of cases [4,5]. The likelihood of complication is directly related to both the experience level and specialty of the proceduralist. Early recognition of malpositioned tubes is the key to preventing risky complications (even death), even more so now that millions of COVID-19 patients are in need of these tubes and lines.

The gold standard for the confirmation of line and tube positions are chest radiographs. However, a physician or radiologist must manually check these chest x-rays to verify that the lines and tubes are in the optimal position. Not only does this leave room for human error, but delays are also common as radiologists can be busy reporting other scans. Deep learning algorithms may be able to automatically detect malpositioned catheters and lines. Once alerted, clinicians can reposition or remove them to avoid life-threatening complications.

The Royal Australian and New Zealand College of Radiologists (RANZCR) is a not-for-profit professional organisation for clinical radiologists and radiation oncologists in Australia, New Zealand, and Singapore. The group is one of many medical organisations around the world (including the NHS) that recognizes malpositioned tubes and lines as preventable. RANZCR is helping design safety systems where such errors will be caught.

In this competition, you’ll detect the presence and position of catheters and lines on chest x-rays. Use machine learning to train and test your model on 40,000 images to categorize a tube that is poorly placed.

The dataset has been labelled with a set of definitions to ensure consistency with labelling. The normal category includes lines that were appropriately positioned and did not require repositioning. The borderline category includes lines that would ideally require some repositioning but would in most cases still function adequately in their current position. The abnormal category included lines that required immediate repositioning.

If successful, your efforts may help clinicians save lives. Earlier detection of malpositioned catheters and lines is even more important as COVID-19 cases continue to surge. Many hospitals are at capacity and more patients are in need of these tubes and lines. Quick feedback on catheter and line placement could help clinicians better treat these patients. Beyond COVID-19, detection of line and tube position will ALWAYS be a requirement in many ill hospital patients.

---

Class Sequence: ETT - Abnormal	ETT - Borderline	ETT - Normal	NGT - Abnormal	NGT - Borderline	NGT - Incompletely Imaged	NGT - Normal	CVC - Abnormal	CVC - Borderline	CVC - Normal	Swan Ganz Catheter Present


# Objective: Multi-Label Binary Classification

In general, a patient's single chest X-ray could present multiple medical conditions, for example, the X-ray can show up Pneumonia and Covid-19 (both are classes), as a result the class labels are not mutually exclusing (unlike multi-class). The same logic is applied in this setting, where the tube can be labelled differently.

<!-- #region -->
# Validation Strategy

We have a few choices, but first we should examine the data for a few factors:

1. Is the data $\mathcal{X}$ imbalanced?
2. Is the data $\mathcal{X}$ generated in a **i.i.d.** manner, more specifically, if I split $\mathcal{X}$ to $\mathcal{X}_{train}$ and $\mathcal{X}_{val}$, can we ensure that $\mathcal{X}_{val}$ has no dependency on $\mathcal{X}_{train}$?

---

We came to the conclusion:


1. Yes, there is quite some imbalanced distribution, in particular, **CVC - Normal**, **ETT - Normal** and **CVC - Borderline** are significantly more than the rest of the classes. Therefore, a stratified cross validation is reasonable. Stratified KFold ensures that relative class frequencies is approximately preserved in each train and validation fold. More concretely, we will not experience the scenario where $X_{train}$ has $m^{+}$ and $m^{-}$ positive and negative samples, but $X_{val}$ has only $p^{+}$ positive samples only and 0 negative samples, simply due to the scarcity of negative samples.

2. In medical imaging, it is a well known fact that most of the data contains patient level repeatedly. To put it bluntly, if I have 100 samples, and according to **PatientID**, we see that the id 123456 (John Doe) appeared 20 times, this is normal as a patient can undergo multiple settings of say, X-rays. If we allow John Doe's data to appear in both train and validation set, then this poses a problem of information leakage, in which the data is no longer **i.i.d.**. One can think of each patient has an "unique, underlying features" which are highly correlated across their different samples. As a result, it is paramount to ensure that amongst this 3255 unique patients, we need to ensure that each unique patients' images **DO NOT** appear in the validation fold. That is to say, if patient John Doe has 100 X-ray images, but during our 5-fold splits, he has 70 images in Fold 1-4, while 30 images are in Fold 5, then if we were to train on Fold 1-4 and validate on Fold 5, there may be potential leakage and the model will predict with confidence for John Doe's images. This is under the assumption that John Doe's data does not fulfill the i.i.d proces

---

With the above consideration, we will use **StratifiedGroupKFold** where $K = 5$ splits. There wasn't this splitting function in scikit-learn at the time of competition and as a result, we used a custom written (by someone else) `RepeatedStratifiedGroupKFold` function and just set `n_splits = 1` to get **StratifiedGroupKFold** (yes we cannot afford to repeated sample, so setting the split to be 1 will collapse the repeated function to just the normal stratified group kfold).

To recap, we applied stratified logic such that each train and validation set has an **equal** weightage of positive and negative samples. We also grouped the patients in the process such that patient $i$ will not appear in both training and validation set.

---

> Data leakage can cause you to have blind confidence on your model. We are also guilty of committing one since we trained our models with the NiH pretrained weights, without taking into consideration if the weights overlap with the training and validation folds information. In other words, we did not check properly if the weights trained on the NiH dataset has information in our RANZCR dataset. Take note this is different from training altogether on the NiH dataset, we are merely using the weights instead of the imagenet weights, which brings to the next point.
<!-- #endregion -->

# Transfer Learning

As we all know, if we train on `imagenet` weights, we may take quite a while to converge, even if we finetune it. The intuition is simple, `imagenet` were trained on many common items in life, and none of them resemble closely to the image structures of X-rays, therefore, the model may have a hard time detecting shapes and details from the X-rays. We can of course unfreeze all the layers and retrain them from scratch, using various backbones, however, due to limited hardware, we decided it is best to use what others have trained. After all, it is much easier to stand on the shoulder of giants like [ammarali](https://www.kaggle.com/ammarali32). Consequently, I conveniently used a set of `pretrained` weights trained specifically on this dataset as a starting point. The weights and ideas can be found **[here](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/215910)**.

We used a few models and found out that `resnet200d` has the best results on this set of training images. The reason we used this is mostly empirical, but using `gradcam` we can see how the model sees the images.


# Preprocessing

Most preprocessing techniques we do in an image recognition competition is mostly as follows:

## Mean and Standard Deviation

- Perform **mean and std** for the dataset given to us. Note that this step may make sense on paper, but empirically, using imagenet's default mean std will always work as well, if not better. Nevertheless, here are the stats:
    - Imagenet on RGB: mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
    - RANZCR using NiH: mean = [0.4887381077884414], std = [0.23064819430546407]),

<!-- #region -->
## Channel Distribution

This is usually done to check for "surprises". More specifically, I remember vividly when participating in Singapore Airline challenge where the classifier recognize weird objects as luggages. After plotting the pixel histogram, we observed that the luggages colors are all of a non-normal distribution, in fact, it is quite scattered. Then it dawned upon us that the classifier is learning the "color" too much, instead of the shape of the luggage. When we grayed out the images, the classifier starts to ignore the noise in the colors, and instead focus on other features like shapes.

We found, and as mentioned also by [Rueben Schmidt](https://www.kaggle.com/reubenschmidt) in [this post](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/224146), there are some images that have black borders around them. I experimented by removed them during both the training process. There was no significant increase on the LB score, even if there was, it is in the 3-4th decimal places, but I noticed my local cv increased, so I think that some noise are removed locally, but not reflected in the test set. Therefore, during inference, I also removed the black borders, which should be the correct approach (learning from mistakes!). In conclusion, there is a small boost in score, if I keep this consistent in both training and inference, I reckon that no surprise factor would pop out.

Here is the code:

```python
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask  = image > 0
image = image[np.ix_(mask.any(1), mask.any(0))]
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
```

> On hindsight for the Singapore Airline project, I now know there is **GradCam**, where we can see how the model is learning, as it will highlight the areas on which the model is focusing on in an image. 
<!-- #endregion -->

## Model Architectures, Training Parameters & Augmentations

> We built-upon fellow Kaggler **Tawara’s** Multi-head model for our best scoring models. In particular, we experimented with the activation functions and dropout rates. We found models with `Swish` activation in the `multi-head` component of the network to perform > best in our experiments. Our best scoring single model is a multi-head model with a `resnet200d` backbone. In particular, one single fold of `resnet200d` gives a private score of 0.970. 
> Another very interesting approach is 3-4 stage training. We did not have time to experiment with the 3-4 stage training as we joined the competition late.

**Model Architectures:**
                    layer = torch.nn.Sequential(
                        SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                        torch.nn.AdaptiveAvgPool2d(output_size=1),
                        torch.nn.Flatten(start_dim=1),
                        torch.nn.Linear(in_features, in_features),
                        self.activation,
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(in_features, out_dim),
                    )
- **Backbone**: `ResNet200D` and `SeResNet152d`
- **Classifier Head:** Separated and Independent **Spatial-Attention Module** and the typical Multi-Layer Perceptron for Target Group (ETT(3), NGT(4), CVC(3), and Swan(1)).
    - **Spatial-Attention Module:** `SpatialAttentionBlock(in_features, [64, 32, 16, 1])`
    - **MLP:**: `Linear -> Swish -> Dropout -> Linear`; It is worth noting after the `Linear` layer, there is a `Sigmoid` layer in this particular setup as we are using `BCEWITHLOGITSLOSS` from PyTorch for numerical stability.
- **Activation:** One thing to note is we used `Swish` in our Classifier Head. Swish is a smooth and non-monotonic function, the latter contrasts when compared to many other activations. I will explain a bit in the next section.
    
- **Pretrained Weight:** [NiH trained](https://www.kaggle.com/ammarali32/startingpointschestx)


## Activation Functions

As we all know, activation functions are used to transform a neurons' linearity to non-linearity and decide whether to "fire" a neuron or not.

When we design or choose an activation function, we need to ensure the follows:

- (Smoothness) Differentiable and Continuous: For example, the sigmoid function is continuous and hence differentiable. If the property is not fulfilled, we might face issues as backpropagation may not be performed properly since we cannot differentiate it.If you notice, the heaviside function is not. We cant perform GD using the HF as we cannot compute gradients but for the logistic function we can. The gradient of sigmoid function g is g(1-g) conveniently

- Monotonic: This helps the model to converge faster. But spoiler alert, Swish is not monotonic.

The properties of Swish are as follows:

- Bounded below: It is claimed in the paper it serves as a strong regularization.
- Smoothness: More smooth than ReLU which allows the model to optimize better, the error landscape, when smoothed, is easier to traverse in order to find a minima. An intuitive idea is the hill again, imagine you traverse down Bukit Timah Hill, vs traversing down Mount Himalaya LOL!!!

```python
# Import matplotlib, numpy and math
import matplotlib.pyplot as plt
import numpy as np
import math

def swish(x):
    sigmoid =  1/(1 + np.exp(-x))
    swish = x * sigmoid
    return swish

epsilon = 1e-20
x = np.linspace(-100,100, 100)
z = swish(x)
print(z)
print(min(z))

plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("Swish(X)")

plt.show()

```

## Model Architecture: Final Activation Layer

> [Sigmoid vs Softmax](https://stats.stackexchange.com/questions/233658/softmax-vs-sigmoid-function-in-logistic-classifier) I've noticed people often get directed to this question when searching whether to use sigmoid vs softmax in neural networks. If you are one of those people building a neural network classifier, here is how to decide whether to apply sigmoid or softmax to the raw output values from your network:

- If you have a multi-label classification problem = there is more than one "right answer" = the outputs are NOT mutually exclusive, then use a sigmoid function on each raw output independently. The sigmoid will allow you to have high probability for all of your classes, some of them, or none of them. Example: classifying diseases in a chest x-ray image. The image might contain pneumonia, emphysema, and/or cancer, or none of those findings.

---

- If you have a multi-class classification problem = there is only one "right answer" = the outputs are mutually exclusive, then use a softmax function. The softmax will enforce that the sum of the probabilities of your output classes are equal to one, so in order to increase the probability of a particular class, your model must correspondingly decrease the probability of at least one of the other classes. Example: classifying images from the MNIST data set of handwritten digits. A single picture of a digit has only one true identity - the picture cannot be a 7 and an 8 at the same time.

---

In the below code we understand that our model's `forward()` call gives us a output `output_logits` of shape (8, 11) if the batch size is 8, and the 11 represents each logit for each of the class. 

- If we apply `Softmax` to this function on `dimension=1`, it simply means we are applying the function each row, from row 1 to 8. Take row 1 for example, the softmax function will squash all the 11 values into a 0-1 range, you can say this is a probability calibration, and the `output_predictions` is also of shape `(8, 11)` but all sums up to 1.

- If we apply `Sigmoid` to this function on `dimension=1`, although PyTorch does not specifiy this because it automatically assumes we are applying sigmoid elementwise, that is to say you cannot simply pass an array of 11 elements to sigmoid function and but we are applying the sigmoid function each row as well. There is a lot of nuance and intricacies here. We take the first row as an example, the first element corresponds to the class **ETT-Abnormal**, when we apply `sigmoid` to this element 0.0762, we get 0.5190, and for the second element class **ETT-Borderline**, we have 0.0877 and when we apply sigmoid, we get 0.5219, so on and so forth for the first row. You should by now observe that they do not sum to 1. This is because each time sigmoid is applied, it is in a **one-vs-all** scenario. Meaning to say, the 0.519 for **ETT-Abnormal** means that **ETT-Abnormal** is treated as the positive class, and the remaining 10 classes are treated as negative class 0. In other words, with 11 elements and sigmoid, we are essentially performing 11 binary classification on the said 11 classes. So 0.519 actually means that the probability of it being class 1 (**ETT-Abnormal**) is 0.519, and the probability of it being NOT class 1 (ALL other classes) is 0.481. The same logic applies to each of the element in the first row. One thing worth noting is that the predictions for row 1 is not **mutually exclusive**, meaning that from the 11 classes, we can have say, **ETT-Abnormal, NGH-Abnormal, CVC-Abnormal** to all have say probability score of 0.9, meaning to say, it is highly likely to be all 3 conditions! This is okay and common in X-Ray imaging.

    - In the table dataframe below, I put them into a dataframe for easy visualization.





classes = ['ETT - Abnormal', 'ETT - Borderline',
           'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
           'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
           'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

```python
import torch

output_logits = torch.tensor([  [ 0.0762,  0.0877,  0.1205, -0.0615, -0.0054,  0.0661,  0.1567, -0.0978, 0.0248, -0.0350,  0.0084],
                                [-0.0196, -0.0729,  0.0534,  0.0307, -0.0428, -0.0016,  0.0013, -0.0247, -0.0094, -0.0424,  0.0192],
                                [-0.0125, -0.0310,  0.0118, -0.1301,  0.0418,  0.0229,  0.0139, -0.0526, 0.0870, -0.0681, -0.0068],
                                [-0.0259, -0.0544, -0.0262,  0.0018,  0.0161, -0.0369, -0.0370, -0.0157, 0.0036, -0.0592,  0.0107],
                                [-0.0366, -0.0695,  0.0740, -0.0353, -0.0363, -0.0019,  0.0085, -0.0144, 0.0129, -0.0470,  0.0043],
                                [-0.0445, -0.0822,  0.0487, -0.0851,  0.0269, -0.0809, -0.0434,  0.0110, -0.0631, -0.0733, -0.0188],
                                [-0.0304,  0.0012,  0.0233, -0.0121, -0.0406, -0.0459, -0.0363,  0.0089,-0.0009, -0.0797, -0.0017],
                                [-0.0415,  0.0787,  0.0283, -0.0617, -0.0526, -0.0016, -0.0409, -0.0481, 0.0583, -0.0810, -0.0050]],
                                dtype=torch.float64, device = 'cuda')
```

```python
sigmoid = torch.nn.Sigmoid()
softmax = torch.nn.Softmax(dim=1)
output_predictions_sigmoid = sigmoid(output_logits)
output_predictions_softmax = softmax(output_logits)
print(output_predictions_sigmoid)
print(output_predictions_softmax)
```

```python
df = pd.DataFrame(data =output_predictions_sigmoid.detach().cpu().numpy(), columns=classes)
display(df)
```

## Model Architecture: Backbone

Empirically, we realized the `ResNet200D` works very well for this particular task. We all asked ourselves why, and it was also discussed by many, but we all agreed that through various experiments, this model seems to consistently outperform their other SOTA counterparts. However, the closest possible paper on [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579).

Of course, to add diversity to our final predictions, we trained one more `SeResNet152d` as well. In general, ensembling models with vastly different architectures may result in a more robust solution. As an example, you can think of each model as a "average learner", and if their structure is different, it may very well so learn information that the other model might miss, hence ensembling them will average out such differences. Later on I will touch upon an ensembling technique called **Forward Ensembling/Selection** in this task, it has since worked well for other similar competitions.

<!-- #region -->
## Model Architecture: Classifier Heads - Multi-Head Approach

- [Reading on self attention in X-ray](https://towardsdatascience.com/self-attention-in-computer-vision-2782727021f6)

- [Image Reference](https://stackoverflow.com/questions/56004483/what-is-a-multi-headed-model-and-what-exactly-is-a-head-in-a-model/56004582)



---

> At the time of competition, both me and my buddy do not have access to premium GPU hardwares, so we cannot afford to experiment with many different models. As we all know, ensembling models will flat out the variance across the model predictions, and may reduce overfitting in general (not always, but in general). The multi-head approach was inspired when reviewing past/current approaches, and of course from Attention is All you need. In addition, this technique is widely used in Object Detection (although really not the same), where in a typical Object Detection problem, we have two heads, Classification and Regression head. The localization problem is split into the two aforementioned heads, where the former is to classify what it is in an image (classification), and the latter head is the **localize** the image in the sense of finding the corrdinates of the bounding boxes around the said image (regression head).

![alt](https://drive.google.com/uc?id=12G-eUVE3lTrBxc8G8mImP3c34EovXD1-) 

Since we have 11 targets in this competition and they can be divided into 4 groups: 

- `ETT`,
- `NGT`,
- `CVC` and
- `Swan`. 

We envision that different groups have different areas in images to focus on. One possible way to leverage this idea is a **multi-head approach**. Multipe groups can share one single CNN backbone but have independent **classifier** heads.

While I cannot fully quantify why multi-head is better than single-head other than empirical results, here are my hypothesis:

Firstly, the backbone (conv layers) are resposible for extracting a feature map from the image, for example, the earlier layers find simple features like shapes, sizes, edges from an image, while the deep conv layers will be of more abstract features in an image. As a result, there is no need to decouple our feature extractor (backbone). With this in mind, let us move on:

**Multi-Label**

This is a multi-label classification problem. The section on the activation functions fully explained the single head version of using sigmoid layer. In fact, it is not uncommon to train N number of heads on a N-class Multi-Label problem. One thing to note is that if your classification head is `Linear` layer only (with BCE loss), then the back gradient propagation is the same whether you train one head, or multiple heads. However, we have non-linear layers in the head, including the `SpatialAttentionBlock`! At the time of writing, I won't say I fully grasp of all the inner workings of an Attention Module across various use cases, but an analogy to aid my understanding is as follows:

> Having taken Learning From Data from Professor Yaser, the inner joke is about the Hypothesis Space. Let me elaborate, given a resnet200D as our hypothesis space $\mathcal{H}$, we aim to find a $h \in \mathcal{H}$ that best represents our true function $f$. Now suppose our learning algorithm $\mathcal{A}$ does a good job in helping us to find such a optimal $h$, it may take time, maybe say 100 epochs before finding it. Now if I break down the problem into 4 parts, each corresponding to a group, and we "aid" the learning algorithm by giving more attention to 4 focused areas, then we might find both a good $h$ that estimate the $f$ well, and may even be faster!

> If the above is too meh for understanding, imagine you are taking an exam in Machine Learning, as we all know, this field is a rabbit hole with never ending topics, let us say that there are 20 topics for you to study for the exam, you are dilligent and does that. But you have limited time and you decided to devote equal time to each topic, the consequence is you may not perform well for the exam due to limited understanding of each topic. Now, if I were to tell you that, hey, out of the 20 topics, can you study these 4 topics, as I think they have a higher chance of coming out, you will likely do better in the exam given that you devoted much more time on those "focused (attention!)" topics.

<!-- #endregion -->

| Single-Head Approach                                                     | Multi-Head Approach                                                      |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| ![alt](https://drive.google.com/uc?id=12AUYKXSH3bb3ENxqbsMjCVDYHvNqcQa9) | ![alt](https://drive.google.com/uc?id=1278TVxrLF8ce-maOGFiJaubMS9lQGgL4) |
courtesy of Tawara

<!-- #region -->
The following code explains this methodology with reference to the above images.

We first note to the readers that typically, if we use a single head approach, where if we were given a problem set $\mathcal{D} = X \times y$, a hypothesis space $\mathcal{H}$ we learn from a learning algorithm $\mathcal{A}$, producing a final hypothesis $g$ (or h, depends on your notation), that predicts as such $g(X_{val}) = y_{\text{val_pred}}$, where each element in $y_{\text{val_pred}}$ corresponds to the class. Think of the basic MNIST example, our prediction vector's first element corresponds to the probability of it being an 0, and so on and so forth (assuming we use soft labels here). 

The change here is after the feature extraction layer (i.e. the feature logits after backbone), instead of just connecting it to a linear head for classification, we instead split the 11 outputs to 4 distinct groups. Each group will go through the head independent of the others, and this may prompt the model to put more **attention** on the independent groups. Finally, we `torch.cat(..,axis=1)` the outputs after they gone through their respective heads to recover the 11 outputs.

```python
model = CustomModel(
    config,
    pretrained=True,
    load_weight=True,
    load_url=False,
    out_dim_heads=[3, 4, 3, 1],
)
# Multi Head
for i, out_dim in enumerate(self.out_dim_heads):
    layer_name = f"head_{i}"
    layer = torch.nn.Sequential(
        SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
        torch.nn.AdaptiveAvgPool2d(output_size=1),
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(in_features, in_features),
        self.activation,
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, out_dim),
    )
    setattr(self, layer_name, layer)

def forward(self, input_neurons):
    """Define the computation performed at every call."""
    if self.use_custom_layers is False:
        output_predictions = self.model(input_neurons)
    else:
        if len(self.out_dim_heads) > 1:
            output_logits_backbone = self.architecture["backbone"](input_neurons)
            multi_outputs = [
                getattr(self, f"head_{i}")(output_logits_backbone)
                for i in range(self.num_heads)
            ]
            output_predictions = torch.cat(multi_outputs, axis=1)
```
<!-- #endregion -->

## Augmentations

We know that augmentation is central in an image competition, as essentially we are adding more data into the training process, effectively reducing overfitting.

In particular, we made use of a different Normalization parameter which is more accustomed to the X-ray pretrained images. Thanks Tawara again! Heavy augmentations are used during Train-Time-Augmentation. But during Test-Time-Augmentation, we merely used a HorizontalFlip with 100% probability, and only used tta_steps=1. 


## Batch Size and Tricks

Due to hardware limitation, we can barely fit in anything more than a `batch_size` of 8.

Quoting from [here](https://arxiv.org/abs/1609.04836):

> It has been observed in practice that when using a larger batch there is a degradation in the quality of the model, as measured by its ability to generalize [...]

> large-batch methods tend to converge to sharp minimizers of the training and testing functions—and as is well known, sharp minima lead to poorer generalization. In contrast, small-batch methods consistently converge to flat minimizers, and our experiments support a commonly held view that this is due to the inherent noise in the gradient estimation.

The above shows that large batch size may `fit` the model too well, as the model will learn features of the dataset in less iterations, and may memorize this particular dataset's features, leading to overfitting and poor generalization. However, too small a batch size causes our convergence to go too slow, empirically, we take 32 or 64 as the ideal batch size in this competition. 

We used both `torch.amp` and `gradient accumulation` to be able to fit more batch sizes. We did not freeze the `batch_norm` layers, which still yielded great results. What we should have done is to experiment more on how to freeze the batch norm layers properly, as I believe that it may help. In the end, we used a batch size of 8 and fit 4 iterations using `gradient accumulation`  and trained a total number of 20 epochs to get a local CV score of roughly 0.969.


## Optimizer, Scheduler and Loss

### Scheduler

The configuration can be seen here. But note that we incorporated `GradualWarmUpScheduler` along with `CosineAnnealingLR`.

From the paper [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677), we learnt about the warmup technique. Although the context of the paper was training under large batch size, we find it helpful even in small batches for the training to converge. 

The basic algorithm is as follows:

1. scheduler_cosine_annealing_warm_restart: Dict = {"T_0": 200,
                                                 "T_mult": 1,
                                                 "eta_min": 0.001,
                                                 "last_epoch": -1,
                                                 "verbose": False}
However, I took quite some time to understand the idea of gradual warmup, I made my understanding [here](https://github.com/reigHns/reighns-pytorch-gradual-warmup-scheduler/blob/master/src/run.py).

We should try `OneCyclePolicy` as detailed by fastai.

### Loss

We should also experiment with `Focal Loss` but seeing negative results from fellow Kagglers, on top with limited resources, we did not try it.

# Ensembling

<!-- #region -->
## Forward Ensembling

We made use of the [Forward Ensembling](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175614) idea from Chris in SIIM-ISIC Melanoma Classification back in August 2020, I modified the code for this specific task. A simple description is as follows, modified from Chris, with more mathematical notations.

1. We start off with a dataset $\mathcal{D} = X \times y$ where it is sampled from the true population $\mathcal{X} \times \mathcal{Y}$.
2. We apply KFold (5 splits) to the dataset, as illustrated in the diagram. 
3. We can now train five different hypothesis $h_{F1}, h_{F2},...,h_{F5}$, where $h_{F1}$ is trained on Fold 2 to Fold 5 and predict on Fold 1, $h_{F2}$ is trained on Fold 1,3,4,5 and predict on Fold 2. The logic follows for all 5 hypothesis.
4. Notice that in the five models, we are predicting on a unique validation fold, and as a result, after we trained all 5 folds, we will have the predictions made on the whole training set (F1-F5). This predictions is called the Out-of-Fold predictions.
5. We then go a step further and calculate the AUC score with the OOF predictions with the ground truth to get the OOF AUC. We save it to a csv or dataframe called **oof_1.csv**, subsequent oof trained on different hypothesis space should be named **oof_i.csv** where $i \in [2,3,...]$.
6. After we trained all 5 folds, we will use $h_{1}$ to predict on $X_{test}$ and obtain predictions $Y_{\text{h1 preds}}$, we then use $h_{2}$ to predict on $X_{test}$ and obtain predictions $Y_{\text{h2 preds}}$, we do this for all five folds and finally $Y_{\text{final preds}} = \dfrac{1}{5}\sum_{i=1}^{5}Y_{\text{hi preds}}$. This is a typical pipeline in most machine learning problems. We save this final predictions as **sub_1.csv**, subsequence predictions trained on different hypothesis space should be named **sub_i.csv** where $i \in [2,3,...]$.
7. Now if we train another model, a completely different hypothesis space is used, to be more pedantic, we denote the previous model to be taken from the hypothesis space $\mathcal{H}_{1}$, and now we move on to $\mathcal{H}_{2}$. We repeat step 1-6 on this new model (Note that you are essentially training 10 "models" now since we are doing KFold twice, and oh, please set the seed of KFold to be the same, it should never be the case that both model comes from different splitting seed for apparent reasons).

---

Here is the key (given the above setup with 2 different models trained on 5 folds):

1. Normally, most people do a simple mean ensemble, that is $\dfrac{Y_{\text{final preds H1}} + Y_{\text{final preds H2}}}{2}$. This works well most of the time as we trust both model holds equal importance in the final predictions.
2. One issue may be that certain models should be weighted more than the rest, we should not simply take Leaderboard feedback score to judge the weight assignment. A general heuristic here is called Forward Selection.
3. (Extract from Chris) Now say that you build 2 models (that means that you did 5 KFold twice). You now have oof_1.csv, oof_2.csv, sub_1.csv, and sub_2.csv. How do we blend the two models? We find the weight w such that `w * oof_1.predictions + (1-w) * oof_2.predictions` has the largest AUC.

```python
all = []
for w in [0.00, 0.01, 0.02, ..., 0.98, 0.99, 1.00]:
    ensemble_pred = w * oof_1.predictions + (1-w) * oof_2.predictions
    ensemble_auc = roc_auc_score( oof.target , ensemble_pred )
    all.append( ensemble_auc )
best_weight = np.argmax( all ) / 100.
```

Then we can assign the best weight like:

```python
final_ensemble_pred = best_weight * sub_1.target + (1-best_weight) * sub_2.target
```

<img src='https://drive.google.com/uc?id=12Mpa_9pTdNYizDCVxq_VxX5qx5j84Y1X' width="500"/>
Coutersy of Chris

---

In this competition, there are two approaches, either maximize the average of the macro AUC score of all the classes, or maximize each column/class separately. It turns out that maximizing the columns separately led to disastrous results (it could be my code and idea is wrong, as ROC is a ranking metric). 
<!-- #endregion -->

# Conclusion

What we could have done better:

- Use more variety of `classifier head` like `GeM`.
- Use more variety of `backbone`.
- Use [Neptune.ai](http://neptune.ai) to log our experiments as soon things start to get messy. Basically MLOps is important!
- Experiment on 3-4 stage training.
- Pseudo Labelling.
- Knowledge Distillation.
- Experiment more on maximizing AUC during ensembles. `rank_pct` etc.

# References

- [Multi-Head Deep Learning Model with Multi-Label Classification](https://debuggercafe.com/multi-head-deep-learning-models-for-multi-label-classification/)

- [AUC Metric on Multi-Label](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff)

- [Sigmoid and Softmax for Multi-Label](https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/)

- [Multi-Label Classification Tutorial](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff)

- [Why we should use Multi-Head in Multi-Label Classification](https://debuggercafe.com/multi-head-deep-learning-models-for-multi-label-classification/)
    - [Follow Up 1](https://debuggercafe.com/multi-label-image-classification-with-pytorch-and-deep-learning/)
    - [Follow Up 2](https://debuggercafe.com/multi-label-fashion-item-classification-using-deep-learning-and-pytorch/)
    - [Follow Up 3](https://debuggercafe.com/deep-learning-architectures-for-multi-label-classification-using-pytorch/)

- [Sigmoid is Binary Cross Entropy](https://stats.stackexchange.com/questions/485551/1-neuron-bce-loss-vs-2-neurons-ce-loss)

- [Attention Blocks in Computer Vision](https://towardsdatascience.com/attention-in-computer-vision-fd289a5bd7ad)

- [Spatial Attention Blocks](https://medium.com/visionwizard/understanding-attention-modules-cbam-and-bam-a-quick-read-ca8678d1c671)

- [Spatial Attention Module](https://paperswithcode.com/method/spatial-attention-module)

- [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

- [Dive Into Deep Learning - Chapter 10: Attention Mechanisms]

- [Gradual Warmup: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)