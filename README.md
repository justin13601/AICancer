
![AICancer](/meta_images/logo.png)

---------  
# AICancer or CaNNcer or think of a name

Inspired by our first APS360 lecture, in which we were shown a pigeon classifying cancer cells [1], we designed a project to use machine learning (ML) to classify complex cancer histopathological images. NAME is an algorithm that differentiates between 5 classes of cancer images: first between colon and lung cell images, then between malignant and benign cells within these categories, and then finally into specific types of malignant cells.

[![Python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://colab.research.google.com/)

![License](https://img.shields.io/github/license/justin13601/AICancer) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/justin13601/AICancer.git/master?filepath=%2Fipynb_testing%2FProject_Notebook.ipynb)


***Disclaimer -*** This is purely an educational project. The information on this repository is not intended or implied to be a substitute for professional medical diagnoses. All content, including text, graphics, images and information, contained on or available through this repository is for educational purposes only.


## Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Project Illustration](#project-illustration)
- [Background & Related Work](#background--related-work)
- [Data & Data Processing](#data--data-processing)
- [Model Architectures](#model-architectures)
- [Baseline Model](#baseline-model)
- [Quantitative Results](#quantitative-results)
- [Qualitative Results](#qualitative-results)
- [Model Evaluation](#model-evaluation)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Project Reflection](#project-reflection)
- [Project Presentation](#project-presentation)
- [References](#references)


## Overview
The current lung and colon cancer diagnosis process requires a doctor to take multiple steps in testing, consulting lung and colon specialists, and receiving secondary opinions before arriving at a complete diagnosis. In our project, a user would acquire and input a histopathological lung or colon image into the model and receive a full diagnosis of cell type, cancerous status, and type of malignancy if applicable. 

ML streamlines the cancer diagnosis process, in which intermediate diagnoses are automated and gives one final output diagnosis. Further, multiple patient diagnoses can be received at once through image batching. Users also do not have to be a doctor; they can be assistants or imaging specialists who relay the output to the primary doctor to analyze. As such, our model acts as a ‘second opinion’ for medical professionals.

<table align="center">
<thead>
  <tr>
    <th align="center">Steps Taken by Medical Professional using ML</th>
    <th align="center">Steps Taken by Medical Professional without ML</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><img src="/meta_images/doctor_with_ml.png" alt="With ML"></td>
    <td><img src="/meta_images/doctor_without_ml.png" alt="Without ML"></td>
  </tr>
  <tr>
    <td colspan="2">· Allows batching of images<br>· Can get multiple diagnoses quickly<br>· Can provide a “second opinion” for doctors<br>· Anyone can use it (i.e. doesn’t have to be the imaging specialist or the doctor, could be an assistant)<br>· Automates a series of decision making into a one-input model<br>· Manual cell classification is done visually which convolutional filters can detect</td>
  </tr>
</tbody>
</table>
<p align="center">
    <b>
        Table 1:
    </b>
    Benefits of Using ML for Cancer Cell Classification
</p>

We chose to classify lung and colon cells specifically as there is an abundance of histopathological cell data for these classes to train and test our model on. Doctors also may not be the ones to extract images from a patient and a classifier that is able to sort images by organ can help reduce misunderstanding in these cases. Further, it allows users to input multiple images in a batch (perhaps each from a different patient) without having to manually sort and remember organ types beforehand. Both organs can be affected by the same type of cancer, adenocarcinoma, which adds complexity to the problem. With an initial organ differentiation step, we hope to minimize these scenarios where cancer is diagnosed properly, but the organ is misclassified. With sufficient available data we would hope to expand classification to other organ types, however we believe the chosen classes illustrate the potential of ML in this situation.


***Features:***
- Uses PyTorch Library
- 4 Convolutional Neural Net Models
- Early Stopping via Checkpoint Files
- Histopathological Images Input in .jpeg or .png Format
- Classification of 5 Classes
- Python 3.6.9


## Dependencies
- NumPy
- PyTorch
- Matplotlib
- Kaggle
- Split-folders
- Torchsummary
- Torchvision
- Pillow


## Quick Start

Install necessary packages:

    pip install -r requirements.txt
    
Edit start.py to include histopathological image path:

    ...
    img = 'dataset\demo\lungn10.png'
    ...
    
Run start.py:

    python start.py

Console output:

    >>> Lung: Benign
    


## Project Illustration
<p align="center"><img src="/meta_images/model_figures.jpg" alt="Project Illustration - Model Figures"></p>
<p align="center">
    <b>
        Figure 1:
    </b>
    Project Illustration - Model Figures
</p>


## Background & Related Work
Machine learning is an evolving technology that is proving to be increasingly useful in cancer diagnosis. Algorithms are already being created for some of the most common forms of the disease: brain cancer, prostate cancer, breast cancer, lung cancer, bone cancer, skin cancer and more [2].

Most recently, a new project received praise for its ability to detect cancerous tumours extremely effectively [3]. The initiative was led by Google Health and Imperial College London in a collaborative effort to use technology to enhance breast cancer screening methods [3]. The algorithm was created using a sample set of 29,000 mammograms, and was tested against the judgement of professional radiologists [3]. When verified against one radiologist, the effectiveness of the algorithm was proven to actually be better, approximately on par with a two-person team [3]. 

The benefits of an algorithm like this one are highly attractive because it offers large savings in time efficiency and can assist in healthcare systems lacking radiologists, such as in the UK [3]. In theory, this algorithm should be able to supplement the opinion of one radiologist to obtain optimal results [3].

Our project has a similar goal of using AI to make decisions about the presence of cancer in scans. The success of this breast cancer algorithm shows that machine learning is capable of completing this task with remarkable results. 


## Data & Data Processing
The data used is from Kaggle [4]. There are 5 classes of data: 2 colon types (benign and malignant: adenocarcinoma) and 3 lung types (benign, non-small cell lung cancer (NSCLC):  malignant - adenocarcinoma (ACA) and malignant - squamous cell carcinoma (SCC)).

<p align="center"><img src="/meta_images/data.png" alt="Data Visualization - Example from Each Class"></p>
<p align="center">
    <b>
        Figure 2:
    </b>
    Data Visualization - Example from Each Class
</p>

This dataset consists of 250 images of each class, which were pre-augmented to 5000 of each class (total of 25 000 images) [4]. We normalized the pixel intensity of the images to the [0,1] range using transforms.Normalize(0.5,0.5,0.5). Images were resized to 224x224 pixels for consistency and to reduce load on our model. Finally, images were transformed to tensors.

<p align="center"><img src="/meta_images/image_processing.png" alt="Data Visualization - Unprocessed vs. Processed"></p>
<p align="center">
    <b>
      Figure 3:
    </b>
    Data Visualization - Unprocessed vs. Processed
</p>

As the dataset used was heavily preprocessed beforehand, our main processing tasks were in splitting and sorting the data. Our classifier is made of 4 linked CNNs, so we needed to ensure that:
1. Each CNN had an appropriate dataset for its classification job
2. All CNNs had a balanced dataset
3. All CNNs were trained on a subset of the same training set
4. All CNNs were individually tested on a subset of the same testing set, which NONE of them had seen before
5. The entire model of linked CNNs was completely tested on a new set that no individual CNN had seen before in training, validation, or individual testing

To achieve this, we split the dataset into training, validation, individual testing, and overall testing sets with the ratio 70 : 15 : 7.5 : 7.5. We created individual model datasets from these, as shown below:

<p align="center"><img src="/meta_images/data_split.jpg" alt="Data Split per Model"></p>
<p align="center">
    <b>
        Figure 4:
    </b>
    Data Split per Model
</p>


## Model Architectures
The architecture will consist of four separate binary CNNs that can be used in sequence or separately to categorize data. All CNNs will take in one preprocessed 224x224 square image and output either a zero or a one depending on the classification determined by the model. The first CNN will distinguish between lung and colon scans. If found to be a lung scan, the image will be passed into CNN #2 which will distinguish between malignant and benign lung cells. If found to be a colon scan, the image will be passed into CNN #3 which will distinguish between malignant and benign colon cells. The fourth CNN will be used in the case of malignant lung scans to classify the type of cancer detected to be either adenocarcinoma or squamous cell.

The CNN architecture was chosen for its invariance properties [5]. Since there is a high level of variance between images, such as differences in cell size, orientation, location, and the amount of cells per image, we require a model that can identify more complex patterns that may be present [5]. 

<table align="center">
<thead>
  <tr>
    <th></th>
    <th align="center">Batch Size</th>
    <th align="center">Learning Rate</th>
    <th align="center"># of Epochs</th>
    <th align="center"># of Convolution Layers</th>
    <th align="center"># of Pooling Layers</th>
    <th align="center"># of Fully Connected Layers</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CNN 1: Lung vs. Colon</td>
    <td align="center">256</td>
    <td align="center">0.001</td>
    <td align="center">14</td>
    <td align="center">2</td>
    <td align="center">2</td>
    <td align="center">2</td>
  </tr>
  <tr>
    <td>CNN 2: Lung Benign vs. Malignant</td>
    <td align="center">150</td>
    <td align="center">0.01</td>
    <td align="center">9</td>
    <td align="center">2</td>
    <td align="center">2</td>
    <td align="center">2</td>
  </tr>
  <tr>
    <td>CNN 3: Colon Benign vs. Malignant</td>
    <td align="center">256</td>
    <td align="center">0.001</td>
    <td align="center">14</td>
    <td align="center">2</td>
    <td align="center">2</td>
    <td align="center">2</td>
  </tr>
  <tr>
    <td>CNN 4: Lung SCC vs. ACA</td>
    <td align="center">64</td>
    <td align="center">0.0065</td>
    <td align="center">13</td>
    <td align="center">4</td>
    <td align="center">1</td>
    <td align="center">2</td>
  </tr>
</tbody>
</table>
<p align="center">
    <b>
        Table 2:
    </b>
    Finalized Model Hyperparameters for Each Convolutional Neural Network
</p>

<p align="center"><img src="/meta_images/cnn1.png" alt="CNN #1 - Lung vs. Colon Architecture"></p>
<p align="center">
    <b>
        Figure 5:
    </b>
    CNN #1 - Lung vs. Colon Architecture
</p>

<p align="center"><img src="/meta_images/cnn2.png" alt="CNN #2 - Lung Benign vs. Malignant Architecture"></p>
<p align="center">
    <b>
        Figure 6:
    </b>
    CNN #2 - Lung Benign vs. Malignant Architecture
</p>

<p align="center"><img src="/meta_images/cnn3.png" alt="CNN #3 - Colon Benign vs. Malignant Architecture"></p>
<p align="center">
    <b>
        Figure 7:
    </b>
    CNN #3 - Colon Benign vs. Malignant Architecture
</p>

<p align="center"><img src="/meta_images/cnn4.jpg" alt="CNN #4 - Lung Malignant SCC vs. ACA Architecture"></p>
<p align="center">
    <b>
        Figure 8:
    </b>
    CNN #4 - Lung Malignant SCC vs. ACA Architecture
</p>


## Baseline Model
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Quantitative Results
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

<table align="center">
<thead>
  <tr>
    <th></th>
    <th>Training Accuracy</th>
    <th>Validation Accuracy</th>
    <th>Testing Accuracy</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CNN 1: Lung vs. Colon</td>
    <td align="center">99.99%</td>
    <td align="center">99.99%</td>
    <td align="center">99.99%</td>
  </tr>
  <tr>
    <td>CNN 2: Lung Benign vs. Malignant</td>
    <td align="center">99.99%</td>
    <td align="center">99.5%</td>
    <td align="center">99.3%</td>
  </tr>
  <tr>
    <td>CNN 3: Colon Benign vs. Malignant</td>
    <td align="center">100%</td>
    <td align="center">94.8%</td>
    <td align="center">96.1%</td>
  </tr>
  <tr>
    <td>CNN 4: Lung SCC vs. ACA</td>
    <td align="center">96.0%</td>
    <td align="center">90.1%</td>
    <td align="center">89.5%</td>
  </tr>
</tbody>
</table>
<p align="center">
    <b>
        Table 3:
    </b>
    Training, Validation, & Testing Accuracies for Each Convolutional Neural Network
</p>

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

CNN #1 - Error/Loss Training Curves | CNN #2 - Error/Loss Training Curves | CNN #3 - Error/Loss Training Curves | CNN #4 - Error/Loss Training Curves 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![CNN1](/meta_images/training_curve_cnn1.png)  |  ![CNN2](/meta_images/training_curve_cnn2.png)  |  ![CNN3](/meta_images/training_curve_cnn3.png) | ![CNN4](/meta_images/training_curve_cnn4.png)
<p align="center">
    <b>
        Table 4:
    </b>
    Error/Loss Training Curves for Each Convolutional Neural Network
</p>


## Qualitative Results
<p align="center"><img src="/meta_images/sample_testing.png" alt="Qualitative Sample Testing"></p>
<p align="center">
    <b>
        Table 5:
    </b>
    Qualitative Sample Testing
</p>

<p align="center"><img src="/meta_images/detailed_results.png" alt="Overall Model Confusion Matrix Analysis"></p>
<p align="center">
    <b>
        Table 6:
    </b>
    Overall Model Confusion Matrix Analysis
</p>

Table 6 shows the model’s classification on the overall testing set, with 373 images of each class. 

All diagonals on the matrix have values above 300, meaning the model has above 80% accuracy on each class. It performs the worst on benign colon scans, which we predicted from our individual sample testing, as CNN 3 did noticeably worse on classifying benign images. Further, the model classifies benign lung images the best. This makes sense as these images only go through CNN 1&2, both which had nearly perfect testing and sample accuracy.

Benign colon scans are often classified as malignant which gives a high false positive rate for colon inputs and causes low accuracy. By comparison, benign lung scans are very rarely classified as malignant, and if they are, they are only classified as ACA. Conversely, only lung ACA cells are mistakenly classified as benign (false negative). Looking at samples of these images shows why: some lung ACA samples have red organelles, similar pink colours, and lots of negative space like benign images, whereas SCC images are overcrowded with dark blue/purple cells.

<p align="center"><img src="/meta_images/dark_imgs.png" alt="Data Visualization: Lung Samples"></p>
<p align="center">
    <b>
        Figure 9:
    </b>
    Data Visualization - Lung Samples
</p>

Lung malignant sub-types are mistaken often, but it is more likely to classify SCC images as ACA. From the above image, one can see some similar features of these two samples. This is even clearer when we look at samples such as lungaca1123, where there is overcrowding and little negative space:

<p align="center"><img src="/meta_images/lungaca1123.png" alt="Data Visualization - lungaca1123.png"></p>
<p align="center">
    <b>
        Figure 9:
    </b>
    Data Visualization - lungaca1123.png
</p>

The final interesting result from our model is that while the lung vs colon classification is nearly perfect, it only failed in classifying colon ACA as lung ACA. This is interesting as colon and lung cells look very different in general, but when looking at some samples, we see that images that do not have defined cell edges look very similar to each other. As the converse isn’t true, there must be some features in colon ACA cells that resemble a lung cell. Overall, we see that lung ACA samples have a variety of features, some of which resemble those of other sub-types, which makes overall classification much more difficult.

<p align="center"><img src="/meta_images/lungaca_vs_colonca.png" alt="Data Visualization - Lung ACA Sample vs. Colon ACA Sample.png
"></p>
<p align="center">
    <b>
        Figure 10:
    </b>
    Data Visualization - Lung ACA Sample vs. Colon ACA Sample.png
</p>


## Model Evaluation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

    
## Discussion
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Ethical Considerations
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Project Reflection
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Project Presentation
<p align="center">
    <a href="https://www.youtube.com/watch?v=2XGbIrBEcU8&feature=youtu.be&fbclid=IwAR3M4EB6KNYCvcTodRVNnIGd4a7XrykxheTOKspXc0Nk4DXqtNi-4Drxnuk">
        <img alt="Project Presentation" width="820" height="461" src="https://res.cloudinary.com/marcomontalbano/image/upload/v1596777447/video_to_markdown/images/youtube--2XGbIrBEcU8-c05b58ac6eb4c4700831b2b3070cd403.jpg">
    </a>
</p>


## References
[1] A. Szöllössi, “Pigeons classify breast cancer images,” BCC News, 20-Nov-2015. [Online]. Available: https://www.bbc.com/news/science-environment-34878151. [Accessed: 10-Aug-2020].

[2] N. Savage, “How AI is improving cancer diagnostics,” Nature News, 25-Mar-2020. [Online]. Available: https://www.nature.com/articles/d41586-020-00847-2. [Accessed: 13-Jun-2020].

[3]	F. Walsh, “AI 'outperforms' doctors diagnosing breast cancer,” BBC News, 02-Jan-2020. [Online]. Available: https://www.bbc.com/news/health-50857759. [Accessed: 13-Jun-2020].

[4] Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). [Dataset]. Available: https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images [Accessed: 18 May 2020].

[5]	S. Colic, Class Lecture, Topic: “CNN Architectures and Transfer Learning.” APS360H1, Faculty of Applied Science and Engineering, University of Toronto, Toronto, Jun., 1, 2020
