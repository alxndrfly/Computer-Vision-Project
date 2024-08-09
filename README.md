# Project in Image Processing and Computer Vision

## Introduction
This project takes a dataset of xray images, one class contains fractured bones, the other class contains non fractured bones. The goal is to train a model capable of classifing xray images into fractured or non fractured. Finally, an app is made to allow users to upload any xray and get an instant prediction.

   - [Bone Fractures](https://drive.google.com/file/d/1WeuxOenviI1_ElW5ISED4MhvR_YFYdmB/view?usp=drive_link): The dataset includes multi-region X-ray images focused on diagnosing bone fractures.


## Project Overview

### 1. First I performed an exploratory data analysis.

Full report can be found in this github repository (Report.pdf).

Based on that analysis i will preprocess images the following way :
- Grayscale convertion of the images
- Resize all images to 150x150 with padding to train the model correctly
- Normalise pixels from [-1, 1]
- Perform data augmentation on the training set only to acheive higher model performance

### 2. Model Training and Performance

Training was performed using the pytorch library and a RTX 4060 gpu.
I will be using convolutional neural networks since they perform well on images.
Adam optimiser, binary cross entropy loss function.

I noticed by increasing epochs and looking at validation loss per epoch that the model
was reaching values as low as 0.05 validation loss.

To optimize the model I decided to include a threshold validation loss number in the
training loop.

That way, when the validation loss would drop below 0.05, the training loop would stop and
I would get a high performing model. This was reached after 40 epochs.

![image](https://github.com/user-attachments/assets/679fb868-bbf5-4f61-932d-7d22197ef2b0)

![image](https://github.com/user-attachments/assets/b6bb1a5b-a988-4723-a489-35a5674fbb8a)


### 3. Conclusions about the model's performance.

Performance :

Quite high at the moment. Now, we could decide to include a threshold in our predictions in
order to maximize recall and be almost sure to include all the true broken bones predicted as
broken bones. ( only 1 image was misclassified as not being a broken bone )
That way, we would never miss a patient with a broken bone but the doctor will spend more
time with cases that do not have a broken bone but are classified as if they did.


Limitations :

Here the model is trained with X Rays only, we tested it and if you predict with a normal pipe
and a broken pipe it will tell you the broken pipe is a broken bone.
Meaning that for our predicting accuracy to be in the same ballpark, we need to predict only
with pictures we know are X Rays of human parts.
The dataset must represent the real-world scenarios well. If the training data is not diverse
enough, the model might not generalize well to unseen data.
Also, the lower the image quality, the worse the X Ray has been done and if the fracture is
very small or unnoticeable the less precise the model will be at predicting and we might not
catch the broken bone.


Selling point :

If a doctor's ability to identify a broken bone from an Xray is lower than 99%, then our model
can help doctors identify patients with broken bones.
The model can also be used as a way for doctors to save time and only focus on individuals
classified as having a broken bone.
Obviously, doctors need to listen to the patient's symptoms and pains in order to include all
patients potentially at risk.
Our model predicts in less than a second if an Xray is a broken bone or not. Allowing doctors
to spend less time looking at the Xray's.

### 4. The app, built using streamlit

This app allows you to upload an xray image and get an instant prediction / classification on wether the bone is fractured or not fractured.

![image](https://github.com/user-attachments/assets/78dfa013-6ad4-4222-b8c3-2b4315b8822c)

![image](https://github.com/user-attachments/assets/783f0c90-7772-4294-9c32-2fb47cbab930)

![image](https://github.com/user-attachments/assets/78dfa013-6ad4-4222-b8c3-2b4315b8822c)

![image](https://github.com/user-attachments/assets/60f842e1-83f3-4660-9c5d-d88034f735e9)

This app was also deployed using streamlit.
