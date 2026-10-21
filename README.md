# NeuroMind

## Run the app

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app\main.py
```

## About MRI model

# 1. Project Overview — In Easy Language

Your project title is “3D MRI-Based ASD-Associated Pattern Classification Using Deep Learning.”

First, let’s understand what this project actually does.

## What is our project about?

Our project uses brain MRI scans and a Deep Learning algorithm called 3D CNN to identify patterns associated with Autism Spectrum Disorder (ASD).

Normally, an MRI scan contains detailed information about the brain. Analyzing these scans manually can be complex. So, we developed a deep learning model that learns patterns from MRI scans and classifies them into two categories:

- ASD-associated: MRI belongs to a participant labelled as ASD in the dataset.
- Control: MRI belongs to a participant labelled as a control in the dataset.

The model learns from previously labelled MRI scans and then predicts a class for a new, preprocessed MRI volume.

3D Brain MRI Scan

Input data

MRI Preprocessing

Normalize intensity and resize the volume

3D CNN (Better3DCNN)

Learns spatial patterns from the brain volume

Prediction and Evaluation

Class probabilities, accuracy and ROC-AUC

Important: This project identifies patterns associated with ASD labels in the dataset. It does not diagnose autism in a person.

## 2. What are the main objectives?

The main objectives of our project are:

1. Use brain MRI data from the ABIDE-II GU dataset.
2. Convert MRI scans into a consistent format suitable for deep learning.
3. Develop a 3D Convolutional Neural Network using PyTorch.
4. Train the model using a GPU in Google Colab.
5. Evaluate the model using accuracy and ROC-AUC.
6. Create a simple interface where a user can upload a preprocessed MRI volume and receive a prediction score.

# 3. Steps We Performed in the Project

Let’s understand the complete workflow in the order we performed it.

Step 1: Dataset Collection

We used the ABIDE-II GU dataset, which contains anatomical brain MRI scans and corresponding participant labels.

- Total participants: 106
- ASD-associated cases: 51
- Control cases: 55

These labels are used as the correct answers during model training.

Step 2: MRI Preprocessing

Raw MRI scans cannot always be fed directly into a deep learning model. So, we prepared them first.

- Intensity normalization: Scaled MRI intensity values between 0 and 1 using percentile-based normalization.
- Resizing: Converted the MRI volume from approximately \\(176 \times 256 \times 256\\) to \\(128 \times 160 \times 160\\).
- Data storage: Saved the processed MRI volumes as `.npy` files.
- Label assignment: Mapped `DX_GROUP = 1` to ASD-associated and `DX_GROUP = 2` to control.

Step 3: Dataset Splitting

We divided the 106 participants into three separate groups.

| Training   | 74 |
| ---------- | -- |
| Validation | 16 |
| Testing    | 16 |

- Training set: Used to teach the model.
- Validation set: Used to monitor performance and help select the model.
- Test set: Used for final evaluation on data not used during training or model selection.

We used a stratified split to maintain class proportions and ensured there was no participant overlap between the sets.

Step 4: Building the Deep Learning Model

We developed a model called Better3DCNN using PyTorch.

It contains four 3D convolutional blocks. The feature channels increase like this:

## 1 → 16 → 32 → 64 → 128

Increasing feature channels across the convolutional blocks

The model learns useful spatial patterns from the three-dimensional brain images.

Step 5: Model Training

We trained the model using PyTorch on Google Colab with a T4 GPU.

During training, the model compares its predictions with the actual labels and adjusts its weights to reduce errors.

Our settings were:

- Optimizer: AdamW
- Learning rate: 0.0001
- Batch size: 2
- Maximum epochs: 30
- Loss function: CrossEntropyLoss
- Early stopping patience: 7 epochs

Training stopped early at epoch 15, based on the early-stopping procedure.

Step 6: Model Evaluation

We evaluated the final model using the untouched test set.

| Metric                  | Result |
| ----------------------- | ------ |
| Test accuracy           | 50%    |
| Test ROC-AUC            | 0.75   |
| Best validation ROC-AUC | 0.746  |

These values tell us how well the model performed. We will explain the meaning of each metric below.

Step 7: Creating the Prediction Interface

We created a simple prototype using Gradio.

The user selects a preprocessed `.npy` MRI volume. The model processes it and returns class probabilities using Softmax. The ASD-associated class probability is displayed as the prediction score.

The possible outputs are “ASD-associated pattern” and “No ASD-associated pattern.”

# 4. Which algorithm did we use, and why?

Our main algorithm is 3D Convolutional Neural Network (3D CNN), specifically our custom Better3DCNN model.

## What is a CNN?

CNN stands for Convolutional Neural Network. It is a deep learning algorithm commonly used to learn features from image data.

For example, when a CNN processes a normal image, it can learn edges, shapes and patterns. In our project, the model learns spatial patterns from brain MRI volumes.

## Why did we use a 3D CNN instead of a 2D CNN?

[Imaging | Cerebral Circulation and Cognition Lab | Cumming School of Medicine | University of Calgary](https://images.openai.com/static-rsc-4/PXAKd984IR7GcoUgsOsQkbHCaIBKgmMnXcWDlKVlO79cYQWKLt4TYBzuIY8wy1mwQEhBLUh7qevnt7ZZv4Uhf26AD4IE4xlCFoHRwWTNFyF7JjHylMh8TvBqEfwz6a90-3Jd1pxHZ5G1x5NlwvPyHet58l8esYl9VM1DZ5LshV0?purpose=inline)

2D CNN

Processes an individual 2D image or slice at a time.

[MRI captures the most detailed view yet of a whole human brain | Science News](https://images.openai.com/static-rsc-4/vieNX3OoXQ-A8IWmWuolFgv417J7B1-Hc5Tr2ATYw8a6_O7R4w98id3QzeiBUJg-F7cpa6NW2_EAIHN9hP0D9vqM5ALeT0mYA-sEad0mVrtmyAiJXXzIbFkr0HfMTGmzrOZbf92uJVf9RgZvmULCAUNXSzXLd1riEvQXOuvVlsk?purpose=inline)

3D CNN

Processes a 3D volume and learns patterns across its spatial dimensions.

Our MRI scans contain three dimensions, so a 3D CNN is suitable because it can learn spatial relationships across neighboring slices and throughout the volume, rather than treating each slice independently.

In simple words: A 2D CNN looks at a flat picture, while a 3D CNN can learn from the brain volume in three dimensions.

## How does our 3D CNN work?

1. Input: The preprocessed 3D MRI volume is given to the model.
2. 3D Convolution: Filters move across the volume to extract useful features.
3. ReLU activation: Introduces non-linearity so the model can learn complex patterns.
4. Max Pooling: Reduces spatial dimensions while retaining strong features.
5. Batch Normalization: Helps stabilize training.
6. Adaptive Average Pooling: Summarizes extracted feature maps into a fixed-size representation.
7. Dropout: Helps reduce overfitting by randomly disabling some activations during training.
8. Linear layer and Softmax: Produce scores and probabilities for the two classes.

These are the components documented in our Better3DCNN architecture. The network learns which combinations of features are useful for separating the dataset's two labelled groups.

# 5. Important algorithm and training concepts

| Term                   | Easy explanation                                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| Deep Learning          | A type of machine learning that uses neural networks with multiple layers.                                  |
| PyTorch                | The Python framework used to build and train our neural network.                                            |
| AdamW                  | The optimizer that updates the model's weights to reduce prediction errors.                                 |
| Learning rate = 0.0001 | Controls the size of weight updates during training.                                                        |
| Batch size = 2         | The model processes two MRI volumes per training batch.                                                     |
| Epoch                  | One complete pass through the training dataset.                                                             |
| CrossEntropyLoss       | Measures how far the model's predicted class scores are from the correct class labels.                      |
| Early stopping         | Stops training when validation performance no longer improves sufficiently, according to the stopping rule. |
| Softmax                | Converts the model's two class scores into probabilities that sum to 1.                                     |

## Why did we use Google Colab and a GPU?

3D MRI volumes contain a large amount of data. Training a 3D CNN can require substantial memory and computation.

Google Colab provided a convenient environment, and the T4 GPU accelerated the model's tensor computations compared with relying only on a CPU.

# 6. How do we measure model performance?

We used two main evaluation metrics: accuracy and ROC-AUC.

Test Accuracy

## 50%

Accuracy measures the proportion of test examples classified correctly.

In our test set of 16 participants, 50% accuracy corresponds to 8 correctly classified participants.

Test ROC-AUC

## 0.75

ROC-AUC measures how well the model ranks the two classes across different classification thresholds.

A value of 0.5 generally indicates chance-level ranking, while 1.0 indicates perfect ranking. Our value of 0.75 suggests some ability to distinguish the classes, but it does not mean the model is 75% accurate.

Important result: Our accuracy is only 50%, so the model's final classification performance needs improvement. The ROC-AUC of 0.75 is a separate measure and does not establish that the model is reliable for diagnosis. Because the test set contains only 16 participants, these results are also uncertain and need further validation.

# 7. Other key points you should know for your viva

## Dataset and file formats

- ABIDE-II GU: The MRI dataset used in this project.
- NIfTI (`.nii.gz`): The original medical imaging file format used to store MRI volumes and associated spatial information.
- NumPy (`.npy`): The format used to save the preprocessed MRI volumes.
- DX_GROUP: The dataset label used to distinguish ASD-associated participants from controls.

## Tools and their purposes

| Tool         | Why we used it                                                              |
| ------------ | --------------------------------------------------------------------------- |
| Python       | Main programming language                                                   |
| PyTorch      | Build and train the 3D CNN                                                  |
| NumPy        | Store and manipulate numerical MRI arrays                                   |
| NiBabel      | Read medical imaging formats such as NIfTI                                  |
| Scikit-learn | Support machine learning utilities, including evaluation and data splitting |
| Google Colab | Run the training notebook with GPU support                                  |
| Gradio       | Build the simple prediction interface                                       |

## What are the limitations of our project?

1. The dataset contains only 106 participants, which is small for training a deep learning model.
2. Test accuracy is 50%, showing that classification performance requires improvement.
3. The model needs testing on larger, independent datasets to assess whether it generalizes to new participants.
4. A prediction score is not a medical diagnosis.
5. Future improvements could include data augmentation, transfer learning, hyperparameter tuning and explainable AI.

## 8. Short viva answer: Explain your project

You can use this answer when your teacher asks, “Explain your project.”

“Our project is titled 3D MRI-Based ASD-Associated Pattern Classification Using Deep Learning. We used the ABIDE-II GU dataset, which contains 106 participants, including 51 ASD-associated cases and 55 control cases. First, we preprocessed the MRI volumes by normalizing their intensity values and resizing them to a common shape. Then, we split the data into training, validation and testing sets. We developed a Better3DCNN model using PyTorch, which uses 3D convolutional layers to learn spatial patterns from brain MRI volumes. We trained the model on Google Colab using a T4 GPU and evaluated it using accuracy and ROC-AUC. The test accuracy was 50%, and the test ROC-AUC was 0.75. Finally, we created a Gradio interface to demonstrate predictions on preprocessed MRI volumes. The project is a research prototype and is not intended for medical diagnosis.”

Remember these five points first:

- Problem: Classify MRI volumes according to ASD-associated and control labels.
- Dataset: ABIDE-II GU, 106 participants.
- Algorithm: Better3DCNN, a 3D Convolutional Neural Network.
- Tools: Python, PyTorch, Google Colab, NumPy, NiBabel and Gradio.
- Results: 50% test accuracy and 0.75 test ROC-AUC.

You can now ask me about any individual part, such as “Explain 3D CNN in more detail,” “What is ROC-AUC?” or “Explain every preprocessing step.” I’ll explain it in the same easy way.

## Behavioral assessment

The behavioral section uses ten questionnaire answers, represented as the
features `A1_Score` through `A10_Score`. Each answer is converted to a number:
Yes is `1` and No is `0`. The training script reads the child dataset from
`data\child_autism_dataset.xlsx` and the adult dataset from
`data\autism_screening.csv`. It trains and saves separate Logistic Regression
models for children and adults.

### Behavioral model training: steps involved

1. **Load the datasets:** The training script loads the child spreadsheet and
   adult CSV file.
2. **Clean duplicate records:** Duplicate rows are removed from each dataset.
3. **Choose inputs and labels:** The ten questionnaire scores are the inputs.
   The `Class/ASD` column is the answer the model learns to predict, encoded
   as `YES = 1` and `NO = 0`.
4. **Split the data:** The script sets aside 20% of the records for testing
   and uses the other 80% for training. It uses a fixed random seed and
   stratifies the split to keep the class proportions similar in both groups.
5. **Train the model:** Logistic Regression learns how the ten answers relate
   to the dataset labels. In simple terms, it estimates which of the two
   labels is more likely for a particular combination of answers.
6. **Evaluate the model:** The script reports test-set accuracy, precision,
   recall, F1 score, a confusion matrix, and five-fold cross-validation
   accuracy. It also calculates the accuracy of a separate simple rule that
   predicts YES when at least 7 of the 10 scores are 1. That rule is a
   comparison; it is not the trained Logistic Regression model.
7. **Save the models:** The trained models are saved as
   `models\child_model.pkl` and `models\adult_model.pkl` so the app can load
   them later without retraining.

### Behavioral prediction in the app

1. The user answers the ten questions on the behavioral assessment page.
2. The app converts Yes/No answers to 1/0 and arranges them in the same
   `A1_Score` to `A10_Score` order used during training.
3. The model returns a predicted class and a probability for the YES class.
   The page shows the result as either “Possible ASD Pattern” or
   “Lower ASD Pattern.”

The prediction code supports ages 4–11 and 18+. The current questionnaire
page loads the child model. The reported probability is a model output, not
a confirmed or medically calibrated probability of autism. The behavioral
result is experimental and is not a diagnosis.

## Facial analysis model

The face-analysis page uses the checkpoints in `models\facial`: ResNet-18
autism and emotion models, a fusion classifier, and out-of-distribution
statistics. Upload a JPG/PNG photo or capture one with the live camera, then
choose an age under 15 to enable **Analyze Face**. The model's ASD likelihood,
emotion scores, and an input-distribution warning are shown in the page and
Results.

Age is self-reported: these checkpoints classify autism-related patterns and
emotions, but do not estimate or verify age. They cannot reliably prevent an
adult from being analyzed if an incorrect age is entered. Facial results are
experimental and are not a diagnosis.

## Combined final score

After behavioral, MRI, and facial analyses are all complete, Results shows a
weighted combined ASD score calculated from their ASD-class probabilities:
MRI 70%, behavioral 15%, and facial 15%. This is an experimental prototype
score, not a calibrated medical probability or a diagnosis.

Uploaded MRI and facial previews are retained in the current Streamlit session
so they remain visible after analysis and when navigating to Results.
Use **Start New Assessment** in the sidebar to clear all current-session
answers, predictions, and uploaded images before another person begins.




#For me to explain them in easy way(MRI)
Haan, bilkul! 😊 Chalo ab hum tumhare project ke important concepts ko bahut easy Hinglish mein, examples ke saath samajhte hain, taaki tumhe viva mein bhi easily explain kar pao.

# 1. What is 3D CNN?

Full form: 3D Convolutional Neural Network

Sabse pehle CNN samjho.

CNN ek deep learning algorithm hai jo images se important features seekhta hai.

For example, agar hum cat ki image dete hain, toh CNN uske ears, eyes aur face ke patterns seekh sakta hai.

Ab tumhare project mein cat ki image nahi, balki 3D brain MRI scan hai.

[Imaging | Cerebral Circulation and Cognition Lab | Cumming School of Medicine | University of Calgary](https://images.openai.com/static-rsc-4/PXAKd984IR7GcoUgsOsQkbHCaIBKgmMnXcWDlKVlO79cYQWKLt4TYBzuIY8wy1mwQEhBLUh7qevnt7ZZv4Uhf26AD4IE4xlCFoHRwWTNFyF7JjHylMh8TvBqEfwz6a90-3Jd1pxHZ5G1x5NlwvPyHet58l8esYl9VM1DZ5LshV0?purpose=inline)

2D CNN

Ek flat image ya MRI slice ko process karta hai. Agar alag-alag slices independently process karein, toh unke beech ki 3D spatial information directly capture nahi hoti.

[Why Humans Are So Smart—And So Stressed Out – SAPIENS](https://images.openai.com/static-rsc-4/0jbTCz-0By5Xta2FBjTAoXBd02BdyDAqcVPatxVDdExMChYfQ2dljlhvcqwQAxmP278Aq213rxiLa7PL6kKnGmk3genV5MdnfNqJ581YyIrafcGFg_uhguez9owOTn-mRDqsf2FPwRMPaGW8qPuUyNpE5GnKnBF5aO66BTQAhfU?purpose=inline)

3D CNN

Brain ke volume ki width, height aur depth par kaam karta hai. Isliye neighboring slices aur unke beech ke spatial patterns ko seekh sakta hai.

### Tumhare project mein kaise kaam karta hai?

1. Hum preprocessed MRI volume input dete hain.
2. CNN us volume mein patterns identify karta hai.
3. Multiple layers milkar useful features seekhti hain.
4. Last layer do classes ke scores generate karti hai.
5. Softmax un scores ko probabilities mein convert karta hai.

Viva answer: “We used a 3D CNN because our input is a three-dimensional brain MRI volume. It can learn spatial features across the width, height and depth of the brain.”

# 2. What is MRI preprocessing?

Preprocessing ka simple meaning hai raw data ko model ke liye ready karna.

Tumhare project mein MRI preprocessing ke 3 important steps hain.

[Magnetic resonance image (MRI) of the brain. MRI scan of human brain axial view with black background.](https://images.openai.com/static-rsc-4/OQElaVHcfNwHroMyJDzHUJEzRuNHa2S4YthtDW9lhHCaaWVvFKQJXChK2lN_CB_XA8yBpq0kbDXrJv4k43du2WaDOM2ESM8Dt2o0hRxe6CXWo1fyF6cv8wybU4x5yFJn1UqhUb60cyipMhA4pngAGH1QRRWawiMykfpB4xpGXzC2gxOIlqZJnMvTaMZPO6p6?purpose=inline)

A. Intensity Normalization

MRI ke pixel/voxel intensity values ko ek common range mein laate hain.

Tumhare project mein values ko 0 se 1 ke beech normalize kiya gaya.

Example: Agar values alag-alag scales par hain, toh unhe common scale par laane se model ke liye data process karna easier hota hai.

[脳 MRI 3D : 正常解剖学 | e-Anatomy](https://images.openai.com/static-rsc-4/kyTsjUBMGev4JoXQStm6LH-dDYNAHhl3JuwX6C8frizs7gmdm0zDV2FwGlqvSeByH7_kML1yfMrnmEi5-UutzDBKpyv4EmD-4prgDb8BIaklOlbWFoMqCyFwMxm4IDqi42jw92vhKsuiqecXLjWxTOuphSOEaOcqZfHdFH9DbQ4?purpose=inline)

B. Resizing

Original MRI volume ka shape approximately \\(176 \times 256 \times 256\\) tha.

Humne ise \\(128 \times 160 \times 160\\) mein convert kiya.

Isse sabhi volumes ka input shape same ho gaya aur computation ka load kam hua.

.npy

C. Data Storage

Preprocessed MRI volumes ko `.npy` format mein save kiya.

Ye NumPy ka file format hai, jisme numerical arrays store kiye ja sakte hain. Baad mein model in arrays ko input ke roop mein use karta hai.

Viva answer: “MRI preprocessing means preparing the raw MRI data before giving it to the model. We normalized the intensity values, resized all volumes to a common shape and saved them as NumPy files.”

# 3. What is ROC-AUC?

Ye tumhare project ka important evaluation concept hai.

Pehle Accuracy samjho.

Maan lo test set mein 16 participants hain aur model ne 8 ko correctly classify kiya.

\\[ \text{Accuracy}=\frac{\text{Correct predictions}}{\text{Total predictions}}\times100 \\]

\\[ =\frac{8}{16}\times100=50\\% \\]

Matlab test set par 16 mein se 8 predictions correct the.

Ab ROC-AUC kya hai?

- ROC ek curve hai jo different classification thresholds par model ki performance compare karta hai.
- AUC ka meaning hai Area Under the Curve.

Easy language mein, ROC-AUC batata hai ki model dono classes ke examples ko alag rank karne mein kitna achha hai.

| ROC-AUC value | Simple meaning                               |
| ------------- | -------------------------------------------- |
| 0.5           | Random ranking jaisa                         |
| 0.75          | Classes ko distinguish karne ki kuch ability |
| 1.0           | Perfect ranking                              |

Tumhare project ka test ROC-AUC 0.75 hai.

Dhyan do: 0.75 ROC-AUC ka matlab 75% accuracy nahi hota. Tumhari actual test accuracy 50% thi.

Viva answer: “Accuracy measures how many predictions are correct, while ROC-AUC measures how well the model distinguishes between two classes across different thresholds.”

# 4. What is Softmax?

Tumhare model ki final layer do classes ke liye scores produce karti hai.

Softmax in scores ko probabilities mein convert karta hai.

Example ke liye, maan lo model ye probabilities deta hai:

ASD-associated

### 70%

Control

### 30%

Illustrative example only — these are not your model's actual prediction results.

Dono probabilities ka total 100% hai.

Aise example mein model ASD-associated class ko higher probability deta hai.

Lekin yaad rakho: ye model ka prediction score hai, diagnosis ya confirmed 70% chance of autism nahi.

Viva answer: “Softmax converts the model's output scores into probabilities for the two classes. We display the ASD-associated class probability in our Gradio prototype.”

# 5. What are Training, Validation and Testing?

Isko exam ki example se samjho.

Training — 74 participants

Jaise student textbook se prepare karta hai, waise model in examples se patterns seekhta hai.

Validation — 16 participants

Jaise practice test se check karte hain ki preparation kaisi hai, waise validation data se model ki performance monitor karte hain.

Testing — 16 participants

Jaise final exam mein unseen questions solve karte hain, waise test data final model ki performance check karta hai.

Humne stratified splitting use ki, jisse classes ka proportion har split mein approximately maintain rahe. Ek participant ke scans different splits mein overlap nahi karte the.

Viva answer: “We divided the dataset into training, validation and testing sets. Training teaches the model, validation helps monitor and select the model, and testing evaluates its final performance on unseen data.”

# 6. What are AdamW, CrossEntropyLoss and Early Stopping?

Ye training ke time use hone wale important concepts hain.

| Concept               | Easy meaning                                                                                               |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| AdamW optimizer       | Model ke weights ko update karta hai taaki errors kam ho sakein.                                           |
| CrossEntropyLoss      | Model ki predictions aur correct class labels ke beech error measure karta hai.                            |
| Learning rate: 0.0001 | Har update mein weights kitne bade steps se change honge, usko control karta hai.                          |
| Batch size: 2         | Model ek training batch mein 2 MRI volumes process karta hai.                                              |
| Epoch                 | Training dataset ka ek complete pass.                                                                      |
| Early stopping        | Validation performance improve na hone par training ko jaldi rok deta hai, according to the stopping rule. |

Tumhare project mein maximum 30 epochs set kiye gaye the, lekin early stopping ke kaaran training epoch 15 par ruk gayi.

Easy example: Agar student baar-baar practice kar raha hai, lekin practice-test results improve nahi ho rahe, toh hum practice ka tareeka review karte hain. Early stopping model training mein unnecessary continuation ko rokne mein help karta hai.

# 7. Quick viva practice

Ab tum check kar sakte ho ki concepts samajh aaye ya nahi.

1\. Why did we use a 3D CNN?

Because MRI is a 3D volume

Because it only works with text

Because it always gives 100% accuracy

Correct! Correct! A 3D CNN learns spatial features across the three dimensions of an MRI volume.

2\. What is the purpose of normalization?

To delete MRI scans

To bring intensity values to a common range

To create the Gradio interface

Correct! Correct! In our project, intensity values were normalized between 0 and 1.

3\. What was our test accuracy?

75%

100%

50%

Correct! Correct! Test accuracy was 50%, while test ROC-AUC was 0.75.

4\. What does Softmax do?

Resizes the MRI volume

Converts class scores into probabilities

Splits the dataset

Correct! Correct! Softmax converts the model's class scores into probabilities.

Try again

Agar tum in concepts ko samajh lete ho, toh tum apne project ke overview, algorithm, preprocessing, training aur evaluation se related basic viva questions confidently answer kar sakte ho.



#Behavioural Model:

Haan, bilkul! 😊 Chalo isko bahut easy Hinglish mein, example ke saath samajhte hain. Ye tumhare NeuroMind project ka Behavioral Assessment Model hai.

## 1. Behavioral Model kya karta hai?

Behavioral model mein hum user se 10 questions poochte hain. Har question ka answer `Yes` ya `No` hota hai.

Computer directly Yes/No answers se model train nahi karta, isliye hum answers ko numbers mein convert karte hain.

- Yes = 1
- No = 0

Example:

| Question | Answer | Number |
| -------- | ------ | ------ |
| Q1       | Yes    | 1      |
| Q2       | No     | 0      |
| Q3       | Yes    | 1      |
| Q4       | Yes    | 1      |
| Q5       | No     | 0      |

Isi tarah hum sabhi 10 questions ke answers ko numbers mein convert karte hain.

In 10 values ko hum `A1_Score` se `A10_Score` tak naam dete hain.

Simple meaning: Hum user ke answers ko numbers mein convert karte hain, taaki machine unhe process kar sake.

## 2. Logistic Regression kya hai?

Logistic Regression ek Machine Learning algorithm hai jo do possible classes mein se kisi ek ko predict karne ke liye use hota hai.

Hamare project mein do labels hain:

- YES
- NO

Socho hum model ko pehle se labelled examples dete hain. Har example mein 10 answers hote hain aur saath mein uska dataset label hota hai.

Model in examples se seekhta hai ki kaun-se answer patterns YES ya NO labels ke saath associated hain.

Jab koi naya user 10 answers deta hai, model un answers ko process karke predict karta hai ki YES ya NO mein se kaunsa label zyada likely hai.

10 Questionnaire Answers

Yes = 1, No = 0

Logistic Regression Model

Learned patterns from labelled data

Prediction

YES or NO + estimated probability

Yaad rakhna: Logistic Regression naam mein Regression hai, lekin iska use binary classification ke liye bhi hota hai.

## 3. Model ko train kaise kiya?

Training ka matlab hai model ko purane examples se sikhana.

Tumhare training script mein ye steps hain:

1. Dataset load kiya: Child dataset Excel file se aur adult dataset CSV file se load kiya.
2. Duplicate rows remove ki: Jo records repeat ho rahe the, unhe remove kiya.
3. Input aur output decide kiya: 10 questionnaire scores inputs hain aur `Class/ASD` target label hai.
4. Labels convert kiye: YES = 1 aur NO = 0.
5. Data split kiya: 80% records training ke liye aur 20% testing ke liye rakhe.
6. Model train kiya: Logistic Regression ne training examples se patterns seekhe.
7. Performance check ki: Accuracy, precision, recall, F1-score, confusion matrix aur 5-fold cross-validation use kiye.
8. Model save kiya: Trained model ko file mein save kiya, taaki application use baad mein load kar sake.

### Child aur Adult models alag kyun hain?

Tumhare training script mein child aur adult datasets ke liye separate models train kiye gaye hain.

Simple words mein, dono datasets se alag-alag models patterns seekhte hain.

Lekin tumhare current questionnaire page mein child model load hota hai. Isliye current page child assessment ke liye set up hai.

## 4. Accuracy, Precision, Recall aur F1-score kya hain?

Ye metrics batate hain ki model ki predictions kaisi hain.

| Metric                  | Easy meaning                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------- |
| Accuracy                | Total predictions mein se kitni correct thi.                                          |
| Precision               | Model ne jitne YES predict kiye, unmein se kitne actual YES labels the.               |
| Recall                  | Actual YES-labelled examples mein se model ne kitne YES identify kiye.                |
| F1-score                | Precision aur recall ka combined measure.                                             |
| Confusion matrix        | Correct aur incorrect predictions ka breakdown.                                       |
| 5-fold cross-validation | Data ko 5 parts mein baantkar, alag-alag folds par model ko train aur evaluate karna. |

Important: Tumhare diye gaye description mein in metrics ke actual numerical results nahi hain. Isliye accuracy ya precision ka koi specific number claim nahi karna chahiye jab tak training output check na ho.

## 5. App use karte waqt kya hota hai?

Maan lo koi user Behavioral Assessment page par aata hai.

1. User 10 questions ke answers deta hai.
2. App Yes ko 1 aur No ko 0 mein convert karta hai.
3. Ye 10 numbers child Logistic Regression model ko diye jaate hain.
4. Model YES/NO class predict karta hai aur YES class ki estimated probability deta hai.
5. App result display karta hai.

Possible displayed results:

- Possible ASD Pattern
- Lower ASD Pattern

Ye application ke screening-style outputs hain, medical diagnosis nahi.

Probability bhi automatically medically verified probability nahi hoti.

## 6. Ek important confusion: 7 YES answers ka rule

Tumhare training script mein ek separate rule bhi hai:

Agar 10 answers mein kam se kam 7 answers YES hain, toh YES predict karo.

Lekin dhyan do — ye Logistic Regression algorithm nahi hai.

- 7-answer rule: Fixed condition hai.
- Logistic Regression: Training data se patterns seekhne wala model hai.

Isliye viva mein dono ko alag explain karna.

## 7. Viva mein project kaise explain karna hai?

Ye simple answer practice kar sakte ho:

“Our behavioral assessment uses ten questionnaire questions. We convert Yes answers into 1 and No answers into 0. These ten values are given to a Logistic Regression model, which learns patterns from previously labelled datasets and predicts either YES or NO. We trained separate models for child and adult datasets and evaluated them using accuracy, precision, recall, F1-score, confusion matrix and five-fold cross-validation. In our current application, the child model is used to display a screening-style result. This is an experimental prototype, not a medical diagnosis system.”

## Quick revision

Khud se ye 4 questions answer karne ki try karo:

1. Yes aur No ko numbers mein kyun convert karte hain?
2. Logistic Regression kya predict karta hai?
3. 80% training aur 20% testing ka kya meaning hai?
4. 7 YES answers wala rule aur Logistic Regression same hain ya different?

Show simple answers


