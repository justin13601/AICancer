
![AICancer](/meta_images/logo.png)

---------  
# AICancer or CaNNcer or think of a name

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

[![Python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://colab.research.google.com/)

![License [1]](https://img.shields.io/github/license/justin13601/AICancer?style=flat-square)

***Disclaimer -*** This is purely an educational project. The information on this repository is not intended or implied to be a substitute for professional medical diagnoses. All content, including text, graphics, images and information, contained on or available through this repository is for educational purposes only.


## Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Project Illustration](#project-illustration)
- [Background & Related Work](#background--related-work)
- [Data Processing](#data-processing)
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
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum.
- Uses PyTorch Library
- 4 Convolutional Neural Net Models
- Early Stopping via Checkpoint Files
- Histopathological Images Input in .jpeg or .png Format
- Classification of 5 Classes
- Python 3.6.9

***About the Dataset***

The dataset contains 25,000 histopathological images with 5 classes. All images are originally 768 x 768 pixels in size and are in jpeg file format. The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 500 non-small cell lung cancer tissue (NSCLC): 250 lung adenocarcinomas and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas), and augmented to 25,000 using the Augmentor package.

The follow are the 5 classes, each with 5,000 images:
- Lung benign tissue
- Lung adenocarcinoma
- Lung squamous cell carcinoma
- Colon adenocarcinoma
- Colon benign tissue


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
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Background & Related Work
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Data Processing
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Model Architectures
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

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
    <td align="center">Blank</td>
    <td align="center">Blank</td>
    <td align="center">Blank</td>
    <td align="center">Blank</td>
    <td align="center">Blank</td>
    <td align="center">Blank</td>
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
    <td align="center">Blank</td>
    <td align="center">Blank</td>
    <td align="center">Blank</td>
    <td align="center">Blank</td>
    <td align="center">Blank</td>
    <td align="center">Blank</td>
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
        Table 1:
    </b>
    Finalized Model Hyperparameters for Each Convolutional Neural Network
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
        Table 2:
    </b>
    Training, Validation, & Testing Accuracies for Each Convolutional Neural Network
</p>

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

CNN #1 - Error/Loss Training Curves | CNN #2 - Error/Loss Training Curves | CNN #3 - Error/Loss Training Curves | CNN #4 - Error/Loss Training Curves 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![CNN1](/meta_images/training_curve_cnn1.png)  |  ![CNN2](/meta_images/training_curve_cnn2.png)  |  ![CNN3](/meta_images/training_curve_cnn3.png) | ![CNN4](/meta_images/training_curve_cnn4.png)
<p align="center">
    <b>
        Table 3:
    </b>
    Error/Loss Training Curves for Each Convolutional Neural Network
</p>


## Qualitative Results
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Model Evaluation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

    
## Discussion
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Ethical Considerations
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

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
[1] The MIT License | Open Source Initiative, 2020. [Online]. Available: https://opensource.org/licenses/MIT. [Accessed: 08-Aug-2020].

[2] N. Savage, “How AI is improving cancer diagnostics,” Nature News, 25-Mar-2020. [Online]. Available: https://www.nature.com/articles/d41586-020-00847-2. [Accessed: 13-Jun-2020].

[3]	F. Walsh, “AI 'outperforms' doctors diagnosing breast cancer,” BBC News, 02-Jan-2020. [Online]. Available: https://www.bbc.com/news/health-50857759. [Accessed: 13-Jun-2020].

[4] Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). [Dataset]. Available: https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images [Accessed: 18 May 2020].

[5]	S. Colic, Class Lecture, Topic: “CNN Architectures and Transfer Learning.” APS360H1, Faculty of Applied Science and Engineering, University of Toronto, Toronto, Jun., 1, 2020
