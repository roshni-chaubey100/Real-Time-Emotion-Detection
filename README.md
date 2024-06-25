# Real-Time Emotion Detection
The project detects human emotions of 7 categories, those are "Happy", "Sad", "Disgust", "Angry", "Neutral", "Surprised" and "Fear".  
The emotion labels are therefore mapped to the traits and micro-traits randomly, referenced from the book "Manwatching: A field guide to human behavior" by Desmond Morris.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/roshni-chaubey100/Real-Time-Emotion-Detection.git
cd Real-Time-Emotion-Detection
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the `emotionDetection.py` script:
```bash
python emotionDetection.py
```
> **NOTE:**  
> Before running the `emotionDetection.py`, You must have the `emotion_detector_model.h5` file.  
> To generate the file, You have to run the `trainDataset.py` script. 'OR'
> Download the [model](emotion_detector_model.h5) with 61% accuracy.
2. The webcam will activate, and the system will start detecting and displaying emotions in real-time.
3. Press 'q' to quit the application.

## Files
* `emotionDetection.py`: Main script for running the emotion detection.
* `TrainDataset.py`: Script for preparing the dataset.
* `mapping.csv`: Contains the mappings for the emotion labels.
* `requirements.txt`: Lists the dependencies required for the project.

## Dataset
The project utilizes a pre-labeled dataset of facial expressions from [Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset "Face Expression Recognition Dataset") to train the model. Ensure the dataset is correctly formatted and placed in the appropriate directory before training.  
> **NOTE:**  
> To run the script `trainDataset.py`, place the downloaded dataset in the current folder, that is parted into `images/train` and `images/validation` folders. 

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt "MIT LICENSE") file for details.

##  
For more detailed information, refer to the project report and presentation included in the repository.  
[Project Report (PDF)](Project%20Report%20(Real-Time-Emotion-Detection).pdf)  
[Project Presentation (PPTX)](Real%20Time%20Emotion%20Detection.pptx)
