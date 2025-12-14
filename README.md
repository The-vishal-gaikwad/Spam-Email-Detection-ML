# Spam Email Detection using Machine Learning

### Developed by **Vishal Gaikwad**

GitHub: [The-vishal-gaikwad](https://github.com/The-vishal-gaikwad)

---

## About The Project

Spam detection is the process of identifying and filtering out unwanted or unsolicited messages, typically emails, sent by spammers or malicious actors. These messages may promote products, attempt phishing attacks, or distribute malware. With the rise in digital communication, effective spam detection has become a critical cybersecurity requirement.

This project implements a **Machine Learning–based Spam Email Detection System** using multiple classification algorithms. The system is trained on a popular Kaggle dataset and is capable of accurately distinguishing between **Spam** and **Ham (Legitimate)** emails. The final trained model is deployed using **Streamlit** for real-time predictions.

---

## Built With

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Streamlit**

Install all required libraries using:

```sh
pip install -r requirements.txt
```

---

## Getting Started

Follow these steps to set up the project locally.

### Installation Steps

1. Clone the repository

   ```sh
   git clone https://github.com/The-vishal-gaikwad/Spam-Email-Detection.git
   ```

2. Install dependencies

   ```sh
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook (`.ipynb` file)

   * After successful execution, a `.pkl` model file will be generated for deployment.

---

## Dataset Description

The project uses the **Email Spam Dataset** from Kaggle.

* **Total Emails:** 5,572
* **Features:**

  * `Message` → Email content
  * `Category` → Spam or Ham

---

## Data Pre-processing

### Steps Performed:

* Dropped unnecessary columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`)
* Converted `Category` into binary values

  * Spam → 0
  * Ham → 1
* Checked and handled null values
* Split data into training and testing sets
* Transformed text data using **TF-IDF Vectorization**

### Feature Extraction

* Used `TfidfVectorizer` with:

  * `stop_words='english'`
  * `lowercase=True`
  * `min_df=1`

---

## Model Training & Evaluation

The dataset is split into training and testing sets. Models are trained using the `fit()` method and evaluated using `predict()`.

### Evaluation Metrics Used:

* Accuracy
* Precision
* Recall
* F1-Score

### Machine Learning Models Implemented:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree Classifier
* Random Forest Classifier
* Stacking Classifier

---

## Model Deployment

The trained model is deployed using **Streamlit** for interactive spam detection.

### Run Deployment

```sh
python Spam Classification Deployment.py
```

This will launch a local web application where users can input email text and receive spam classification results instantly.

---

## Contributing

Contributions are welcome and appreciated!

Steps to contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request



---



---

## Acknowledgements

* Developed and maintained by **Vishal Gaikwad**
* Kaggle for providing the Email Spam dataset
* Open-source contributors of NumPy, Pandas, Scikit-learn, and Streamlit

---

If you found this project helpful, feel free to connect and collaborate.
