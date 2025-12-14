# Spam Email Detection using Machine Learning

## About the Project
Spam detection is the process of identifying and filtering unsolicited or harmful emails.
Such emails may promote products, attempt phishing attacks, or distribute malware.
With the rapid growth of digital communication, effective spam detection has become
a critical cybersecurity requirement.

This project implements a Machine Learning–based Spam Email Detection System that
classifies emails as Spam or Ham (legitimate). The system is trained on a public
Kaggle dataset and deployed using Streamlit for real-time predictions.

---

## Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

Install all required libraries:
```bash
pip install -r requirements.txt
Getting Started
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/The-vishal-gaikwad/Spam-Email-Detection.git
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook (.ipynb)

After successful execution, a trained .pkl model file will be generated for deployment.

Dataset Description
Source: Kaggle Email Spam Dataset

Total Emails: 5,572

Features:

Message – Email content

Category – Spam or Ham

Data Pre-processing
Removed unnecessary columns (Unnamed: 2, Unnamed: 3, Unnamed: 4)

Converted category labels to binary values

Spam → 0

Ham → 1

Checked and handled missing values

Split data into training and testing sets

Applied TF-IDF vectorization to text data

TF-IDF Settings:

stop_words = 'english'

lowercase = True

min_df = 1

Model Training & Evaluation
The dataset was split into training and testing sets. Models were trained using the
fit() method and evaluated using predict().

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Models Implemented:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree Classifier

Random Forest Classifier

Stacking Classifier

Model Deployment
The final trained model is deployed using Streamlit for interactive spam detection.

Run the application:

bash
Copy code
python Spam Classification Deployment.py
Users can input email text and receive real-time spam classification results.

My Role
Data preprocessing and cleaning

Feature extraction using TF-IDF

Model training and evaluation

Model deployment using Streamlit

Acknowledgements
Kaggle for providing the Email Spam dataset

Open-source contributors of NumPy, Pandas, Scikit-learn, and Streamlit

Author
Vishal Gaikwad
GitHub: https://github.com/The-vishal-gaikwad
