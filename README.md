# NeuroMind

## Run the app

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app\main.py
```

## MRI model input

MRI inference uses `models\better_3dcnn_asd.pth`. Upload the preprocessed NumPy
volume saved by the Colab pipeline as a `.npy` file. It must contain real,
finite numeric values and have shape `(128, 160, 160)` in `(D, H, W)` order.
The app uses the saved values as-is and does not rescale or normalize them.
Raw NIfTI files and 2D image files are not accepted because they do not match
the checkpoint's training input.

The model's class mapping is class 0 = Control and class 1 = ASD. The result
and its softmax confidence are shown on the MRI page and included in Results.
This is an experimental screening output, not a clinical diagnosis.

### MRI analysis: steps involved

1. **Prepare the scan:** The MRI must first be processed into a 3D NumPy volume
   by the separate Colab preprocessing pipeline. The app does not convert a
   raw scan into this format.
2. **Upload and validate:** Upload the resulting `.npy` file on the MRI page.
   The app checks that it has shape `(128, 160, 160)` and contains finite,
   real numeric values.
3. **Preview:** The app displays the middle slice of the volume as a visual
   preview. This preview is not itself the model input; the model uses the
   full 3D volume.
4. **Analyze:** A 3D convolutional neural network (3D CNN) processes the
   volume. In everyday terms, it looks for patterns across the scan's depth
   as well as its height and width. The network has convolution, normalization,
   and pooling layers, followed by a classifier that chooses between two
   classes.
5. **Show the result:** The app reports the predicted class and its model
   confidence. The saved model maps class 0 to Control and class 1 to ASD.

The repository contains the MRI model architecture and code for loading the
saved checkpoint and making predictions. It does not include the complete
MRI training script, training dataset details, or the Colab preprocessing
steps. Therefore, do not assume that an arbitrary MRI scan can be used: it
must be prepared in exactly the format expected by this checkpoint. The
prediction is experimental and is not a diagnosis.

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
