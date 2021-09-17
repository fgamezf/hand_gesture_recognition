# Hand Gesture Recognition
You can train and create your custom model for hand gesture recognition based on <strong>OpenCV</strong> and <strong>MediaPipe</strong>.

<strong>Requirements:</strong>
- OpenCV
- MediaPipe
- Scikit-learn

<strong>Install:</strong>
<br/>Under the root directory of this project create both directories:
- ./output
- ./model

<strong>Description</strong>:
- main.py - (datasets) Generate CSV file with hand landmarks (./output directory)
- train.py - Create and train a model and export it (./model directory)
-            At least two different classes is required in dataset input file.
- predict.py - Test the model.
