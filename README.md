
![AICancer](/meta_images/logo.png)

---------  
# AICancer or CaNNcer or think of a name

Inspired by our first APS360 lecture, in which we were shown a pigeon classifying cancer cells [1], we designed a project to use machine learning (ML) to classify complex cancer histopathological images. NAME is an algorithm that differentiates between 5 classes of cancer images: first between colon and lung cell images, then between malignant and benign cells within these categories, and then finally into specific types of malignant cells.

[![Python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://colab.research.google.com/)

![License](https://img.shields.io/github/license/justin13601/AICancer) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/justin13601/AICancer.git/master?filepath=%2Fipynb_testing%2FProject_Notebook.ipynb)


***Disclaimer:*** This is purely an educational project. The information on this repository is not intended or implied to be a substitute for professional medical diagnoses. All content, including text, graphics, images and information, contained on or available through this repository is for educational purposes only.

(2486 Words)


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
- [Project Presentation](#project-presentation)
- [References](#references)


## Overview
The current lung and colon cancer diagnosis process requires a doctor to take multiple steps in testing, consulting lung and colon specialists, and receiving secondary opinions before arriving at a complete diagnosis. In our project, a user would acquire and input a histopathological lung or colon image into the model and receive a full diagnosis of cell type, cancerous status, and type of malignancy if applicable. 

ML streamlines the process by automating intermediate steps to give one complete output diagnosis. Users also do not have to be a doctor; they can be assistants who relay the output to the primary doctor to analyze. As such, our model acts as a ‘second opinion’ for medical professionals.

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

We chose to classify lung and colon cells as there is an abundance of data to train and test our model on. Doctors also may not be the ones to extract images from a patient and a classifier that can sort images by organ can reduce misunderstanding in these cases. Further, users can input multiple images in a batch (perhaps each from a different patient) without having to manually sort and remember organ types beforehand. Both organs can also be affected by adenocarcinoma, which adds complexity to the problem. With an organ differentiation step, we hope to minimize scenarios where cancer is diagnosed properly, but the organ is misclassified. With sufficient data we hope to expand classification to other organs, however we believe the chosen classes illustrate the potential of ML in this situation.


***Features:***
- Uses PyTorch Library
- 4 Convolutional Neural Network Models
- Early Stopping via Checkpoint Files
- Histopathological Images Input in .jpeg or .png Format
- Classification of 5 Cancer Cell Classes
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
ML is an evolving technology that is proving to be increasingly useful in cancer diagnosis. Algorithms are being created for some of the most common forms of the disease: brain cancer, prostate cancer, breast cancer, lung cancer, bone cancer, skin cancer and more [2].

Most recently, a new project received praise for its ability to detect cancerous tumours extremely effectively [3]. The initiative was led by Google Health and Imperial College London in an effort to use technology to enhance breast cancer screening methods [3]. The algorithm was created using a sample set of 29,000 mammograms, and was tested against the judgement of professional radiologists [3]. When verified against one radiologist, the effectiveness of the algorithm was proven to be better, on par with a two-person team [3]. 

The benefits of an algorithm like this one are highly attractive because it offers savings in time efficiency and can assist in healthcare systems lacking radiologists [3]. In theory, this algorithm should be able to supplement the opinion of one radiologist to obtain optimal results [3].

Our project has a similar goal of using AI to make decisions about the presence of cancer in scans. The success of this breast cancer algorithm shows that ML is capable of completing this task with remarkable results. 


## Data & Data Processing
The data used is from Kaggle [4]. There are 5 classes of data: 2 colon types (benign and malignant: adenocarcinoma) and 3 lung types (benign, malignant - adenocarcinoma (ACA) and malignant - squamous cell carcinoma (SCC).

<p align="center"><img src="/meta_images/data.png" alt="Data Visualization - Example from Each Class"></p>
<p align="center">
    <b>
        Figure 2:
    </b>
    Data Visualization - Example from Each Class
</p>

This dataset consists of 250 images of each class, which were pre-augmented to 5,000 of each class (total of 25,000 images) [4]. We normalized the pixel intensity of the images to the [0,1] range and images were resized to 224x224 pixels for consistency and to reduce load on our model. Finally, images were transformed to tensors.

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
The architecture consists of four separate binary CNNs. All CNNs take in one preprocessed 224x224 square image and output either a zero or a one depending on the classification determined by the model. The first CNN distinguishes between lung and colon scans. If found to be a lung scan, the image is passed into CNN #2 which distinguishes between malignant and benign lung cells. If found to be a colon scan, the image is passed into CNN #3 which distinguishes between malignant and benign colon cells. The fourth CNN is used to classify the type of lung cancer: adenocarcinoma or squamous cell.

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

A Random Forests Classifier was the selected baseline. 1,000 estimators were used to train over 7,000 images from the training set. Next, 1,000 images were taken from the validation set and run through the classifier. The model achieved an 80.6% accuracy. This model showed several weaknesses, including classifying 19.7% of colon ACA images as benign, and classifying 25.0% of lung SCC images as lung ACA. Overall, differentiation between the two cancerous lung subtypes was often confused, and high false negative rates makes this model unsuitable for use.

<p align="center"><img src="/meta_images/baseline_matrix.png" alt="Baseline Model Confusion Matrix Analysis"></p>
<p align="center">
    <b>
        Table 3:
    </b>
    Baseline Model Confusion Matrix Analysis
</p>


## Quantitative Results

<p align="center"><img src="/meta_images/detailed_results.png" alt="Overall Model Confusion Matrix Analysis"></p>
<p align="center">
    <b>
        Table 4:
    </b>
    Overall Model Confusion Matrix Analysis
</p>

Performance of the model was reasonably good overall, with optimal results in lung benign, lung adenocarcinoma and colon adenocarcinoma scan classes. The overall model accuracy was 91.05%. The model had low error in differentiating lung and colon images. Accuracy between lung malignant and benign was also high. Since individual CNNs were used for each step of the classification process, each one had different hyperparameters and architecture that best fit the classification job.

The accuracies of the more general convolutional neural networks (CNN 1 and CNN 2) are very high, and most error occurred in the periphery CNNs (CNN 3 and CNN 4). This ensured that minimal error propagation occurred.

False negative results for both colon and lung were also low, indicating that the model would rarely ignore a cancerous scan. This is important, as false negative results could potentially cause the patient to have a false sense of security, and result in them seeking treatment much later when the cancer is much more serious. Our model avoids this crucial error.

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
        Table 5:
    </b>
    Training, Validation, & Testing Accuracies for Each Convolutional Neural Network
</p>

Drawbacks include a tendency to give a false positive result on colon images. The model frequently misdiagnosed healthy colon scans as having cancer. This could result in a healthy patient getting an additional scan or biopsy, which could be expensive or uncomfortable. Although this is less serious than false negative classification, it could have detrimental effects.

Additionally, the model occasionally confused the cancerous lung subtypes, although significantly less frequently than the baseline model. This could require the doctor to consult with a specialist or lead the patient to pursue a treatment which is not appropriate. This is why this model is only intended to be used with a doctor at this stage; although the results obtained are very favourable, it is not a replacement for a medical professional.

CNN #1 - Error/Loss Training Curves | CNN #2 - Error/Loss Training Curves | CNN #3 - Error/Loss Training Curves | CNN #4 - Error/Loss Training Curves 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![CNN1](/meta_images/training_curve_cnn1.png)  |  ![CNN2](/meta_images/training_curve_cnn2.png)  |  ![CNN3](/meta_images/training_curve_cnn3.png) | ![CNN4](/meta_images/training_curve_cnn4.png)
<p align="center">
    <b>
        Table 6:
    </b>
    Error/Loss Training Curves for Each Convolutional Neural Network
</p>


## Qualitative Results
<p align="center"><img src="/meta_images/sample_testing.png" alt="Qualitative Sample Testing"></p>
<p align="center">
    <b>
        Table 7:
    </b>
    Qualitative Sample Testing
</p>

In Table 7, we see that the model performs the worst on benign colon scans, which we predicted, as CNN 3 did worse on classifying benign images in sample testing. Further, the model classifies benign lung images the best. This makes sense as these images only go through CNN 1 & 2 which had nearly perfect testing and sample accuracy.

Benign colon scans are often classified as malignant, giving a high false positive rate for colon inputs and causes low accuracy. By comparison, benign lung scans are rarely classified as malignant, and if they are, they are only classified as ACA. Conversely, only lung ACA cells are mistakenly classified as benign (false negative). Looking at samples of these images shows why: some lung ACA samples have red organelles, similar pink colours, and negative space like benign images, whereas SCC images are overcrowded with dark blue cells.

<p align="center"><img src="/meta_images/dark_imgs.png" alt="Data Visualization: Lung Samples"></p>
<p align="center">
    <b>
        Figure 9:
    </b>
    Data Visualization - Lung Samples
</p>

Lung malignant sub-types are mistaken often, but it is more likely to classify SCC images as ACA. This is justified when looking at the following sample, where there is overcrowding and little negative space:

<p align="center"><img src="/meta_images/lungaca1123.png" alt="Data Visualization - lungaca1123.png"></p>
<p align="center">
    <b>
        Figure 10:
    </b>
    Data Visualization - lungaca1123.png
</p>

Although lung vs. colon classification is nearly perfect, our model failed on several occasions and classified colon ACA as lung ACA. Colon and lung cells look very different in general, but when looking at some samples, those that do not have defined cell edges look similar. As the converse isn’t true, there must be some features in colon ACA cells that resemble a lung cell. 

<p align="center"><img src="/meta_images/lungaca_vs_colonca.png" alt="Data Visualization - Lung ACA Sample vs. Colon ACA Sample.png
"></p>
<p align="center">
    <b>
        Figure 11:
    </b>
    Data Visualization - Lung ACA Sample vs. Colon ACA Sample.png
</p>


## Model Evaluation
To determine the model performance on new data, a holdout set of 10 images was used. Due to the nature of this problem, it was difficult to acquire completely new data, so we were restricted to a subset of the original images (holdout set). As our model performed well on these images, it is capable of generalizing to never before seen images, provided they are of the same style.

<p align="center"><img src="/meta_images/evaluation_1.png" alt="Classification of Dataset Images"></p>
<p align="center">
    <b>
        Table 8:
    </b>
    Classification of Dataset Images
</p>

A concern with using images from the same dataset is that the dataset had been augmented from 1250 original images. As we cannot identify augmented images, it is likely that testing images were not completely unique. This limits our ability to determine the models performance on completely new data, as although augmentation is sufficient to ensure the model doesn’t recognise the image, the style and features of the images are very similar.

To further validate our model, we input 1 image of each class found on Google Images [6][7][8][9]. It correctly classified 2 of the 5 samples: the lung and colon benign images. The model correctly differentiated between lung and colon images 80% of the time, but struggled with the individual classes. As it was difficult to find credible images, we were unable to determine the method through which the scans were taken and prepared. However, it is likely that the process differed from the one used to prepare the images in our dataset, which can be visually confirmed by the significant differences in these images from our original ones.

<p align="center"><img src="/meta_images/evaluation_2.png" alt="Classification of Google-Sourced Images"></p>
<p align="center">
    <b>
        Table 9:
    </b>
    Classification of Google-Sourced Images
</p>

Performance on correctly prepared images showed that the model could classify new data, provided the new data was prepared in the same method as the data in the dataset was prepared. Our model was trained on one dataset so it has learned to identify only one method of scan acquisition and processing, which limits performance. However, the architecture and concepts discussed could be generalized if more credible data is made available.

    
## Discussion
Using ML to complement medical diagnoses is a complicated and risky process. While false-positive results can induce further error, false-negative predictions prove fatal to a patient’s health and wellbeing. For example, in the case of CNN #2 and #3, incorrectly classifying a malignant image as benign in practice could be catastrophic to cancer survivorship.

As is the case with most ML models, accuracy is the focal point. Using a test dataset of 1865 images, the algorithm achieved 91.05% accuracy - a successful result when compared with the baseline. However,  8.95% of the test set, surmounting to a total of 167 images, were misclassified, which is extremely significant in medicine.

Because of the significance of even the tiniest  error, it is important to note that at this stage any ML model should only be used in addition to a medical professional’s expertise. Perhaps when our algorithm is used in conjunction with the knowledge of doctors, we could reduce misdiagnoses to a true 0%.

<p align="center"><img src="/meta_images/ai_medicine.jpg" alt="AI in Medicine"></p>
<p align="center">
    <b>
        Figure 12:
    </b>
    AI in Medicine [10]
</p>

Given our unique approach to the problem, it is important to address the benefits of an algorithm with 4 binary neural networks rather than one traditional multi-class CNN. As part of our project investigation, we designed a multi-class classifier to justify our decision.

We noticed several shortcomings of the multi-class classifier: notably, it's memory and time intensive training (given that it would work with significantly more images divided into 5 classes), as well as lower accuracies (having only achieved 65% initially). Having 4 individual binary networks allows for precise tuning at each stage of the algorithm. In addition, having a high-accuracy model that differentiates between organs as the first step significantly reduces misclassification between cancers of similar types (Lung ACA vs. Colon ACA).

A tree of models also permits flexibility for future development. Perhaps CNN 1 could be modified to include the classification of other key organs of the body, or CNN 4 could include small cell lung cancer (SCLC) in addition to NSCLC images (ACA and SCC). Finally, multiple models allow users to selectively “enable” classification functions to best cater their specific needs.
 
Through the development of this repository, we’ve realized the impact of having preprocessed data. While prepared data may be perfect for training or validating a model, that same model would not perform as well on data that is processed differently (prepared with different dyes etc.). In addition, aggregate accuracy values are often misleading, and false negative/positive values are a more precise way to identify weaknesses in a model. Lastly, we’ve acknowledged that a traditional approach to a problem is not always the best one, as in our case separate binary models performed better than a multi-class classifier.


## Ethical Considerations
Ethical consequences of this algorithm are far-reaching, as the data is highly sensitive and the output can directly impact patients. The scans used to train, validate and test this model must be anonymous and consent must be obtained from all patients before the scans are used. 

This algorithm should not be blindly trusted to make serious health diagnoses without the consultation of an experienced doctor. Trusting an algorithm to make health decisions is unethical, as it has limited context in which to make decisions (i.e. one scan only, limited to only those types of cancer) and will have limited accuracy. 

Once the algorithm is ready to use in a clinical or research setting, doctors should be provided with information on how it works and made aware of the risks. This will prevent professional judgement from becoming overly dependent on the program.

Ultimately, there is no doubt that we are still long ways away from being solely dependent on ML in the medical world, but with each project we stride closer to revolutionizing our health care.


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

[6] Histology of the Lung. Youtube, 2016.

[7] J. Stojšić, “Precise Diagnosis of Histological Type of Lung Carcinoma: The First Step in Personalized Therapy,” Lung Cancer - Strategies for Diagnosis and Treatment, 2018.

[8]	V. S. Chandan, “Normal Histology of Gastrointestinal Tract,” Surgical Pathology of Non-neoplastic Gastrointestinal Diseases, pp. 3–18, 2019.

[9]	Memorang, “Colon Cancer (MCC Exam #3) Flashcards,” Memorang. [Online]. Available: https://www.memorangapp.com/flashcards/92659/Colon_Cancer/. [Accessed: 10-Aug-2020].

[10] J. Voigt, “The Future of Artificial Intelligence in Medicine,” Wharton Magazing. [Online]. Available: https://magazine.wharton.upenn.edu/digital/the-future-of-artificial-intelligence-in-medicine/. [Accessed: 10-Aug-2020].
