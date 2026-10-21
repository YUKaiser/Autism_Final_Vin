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
