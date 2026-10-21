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
