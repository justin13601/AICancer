
![AICancer](/meta_images/logo.png)

---------  
# AICancer or CaNNcer or think of a name

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

[![Python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://colab.research.google.com/)

![License](https://img.shields.io/github/license/justin13601/AICancer?style=flat-square)

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
- List features maybe?
- Number of models?
- Histogram input in .jpeg or .png format
- Uses PyTorch
- CNN Models
- Model flow?
- Checkpoint files?
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
    
Edit start.py to include histogram image path:

    ...
    img = 'dataset\demo\lungn10.png'
    ...
    
Run start.py:

    python start.py

Output:

    >>> Lung: Benign
    


## Project Illustration
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Background & Related Work
Machine learning is an evolving technology that is proving to be increasingly useful to the field of cancer diagnosis. Algorithms are already being created for some of the most common forms of the disease: brain cancer, prostate cancer, breast cancer, lung cancer, bone cancer, skin cancer and more [2].

Most recently, a new project received praise for its ability to detect cancerous tumours extremely effectively [3]. The initiative was led by Google Health and Imperial College London in a collaborative effort to use technology to enhance breast cancer screening methods [3]. The algorithm was created using a sample set of 29,000 mammograms, and was tested against the judgement of professional radiologists [3]. When verified against one radiologist, the effectiveness of the algorithm was proven to actually be better, approximately on par with a two-person team [3]. 

The benefits of an algorithm like this one are highly attractive because it offers large savings in time efficiency and can assist in healthcare systems lacking radiologists, such as in the UK [3]. In theory, this algorithm should be able to supplement the opinion of one radiologist to obtain optimal results [3].

Our project has a similar goal of using AI to make decisions about the presence of cancer in scans. The success of this breast cancer algorithm shows that machine learning is capable of completing this task with remarkable results. 



## Data Processing
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Model Architectures
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Baseline Model
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Quantitative Results
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

CNN #1 - Error/Loss Training Curves | CNN #2 - Error/Loss Training Curves | CNN #3 - Error/Loss Training Curves | CNN #4 - Error/Loss Training Curves 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![CNN1](/meta_images/training_curve_cnn1.png)  |  ![CNN2](/meta_images/training_curve_cnn2.png)  |  ![CNN3](/meta_images/training_curve_cnn3.png) | ![CNN4](/meta_images/training_curve_cnn4.png)


## Qualitative Results
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Model Evaluation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.

    
## Discussion
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Ethical Considerations
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Project Reflection
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis id ullamcorper augue, eget euismod lorem. Nunc scelerisque massa sit amet dapibus rutrum. Donec sit amet sapien ante. Nam non dapibus eros. Duis condimentum nisi non rutrum finibus. Donec venenatis lorem a ultrices molestie. Nunc a mauris aliquam augue tincidunt ultrices ac in mi. Etiam vitae efficitur erat. Praesent accumsan augue et lectus congue aliquam.


## Project Presentation
[![Project Presentation](https://res.cloudinary.com/marcomontalbano/image/upload/v1596777447/video_to_markdown/images/youtube--2XGbIrBEcU8-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=2XGbIrBEcU8&feature=youtu.be&fbclid=IwAR3M4EB6KNYCvcTodRVNnIGd4a7XrykxheTOKspXc0Nk4DXqtNi-4Drxnuk "Project Presentation")


## References
[2] N. Savage, “How AI is improving cancer diagnostics,” Nature News, 25-Mar-2020. [Online]. Available: https://www.nature.com/articles/d41586-020-00847-2. [Accessed: 13-Jun-2020].

[3]	F. Walsh, “AI 'outperforms' doctors diagnosing breast cancer,” BBC News, 02-Jan-2020. [Online]. Available: https://www.bbc.com/news/health-50857759. [Accessed: 13-Jun-2020].

[4] Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). [Dataset]. Available: https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images [Accessed: 18 May 2020].

[5]	S. Colic, Class Lecture, Topic: “CNN Architectures and Transfer Learning.” APS360H1, Faculty of Applied Science and Engineering, University of Toronto, Toronto, Jun., 1, 2020
