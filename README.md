Machine Learning Portfolio Project 

This repository contains the completed tasks for a machine learning and data science assignment, covering classical supervised learning, deep learning, and natural language processing (NLP). The code is designed to be executed within a Python virtual environment on an Ubuntu machine, preferably using a Jupyter Notebook within VS Code.

 Project Overview

The project is divided into three distinct tasks:

Classical ML: Training and visualizing a Decision Tree Classifier on the Iris dataset.

Deep Learning: Building and evaluating a Convolutional Neural Network (CNN) on the MNIST handwritten digits dataset.

Natural Language Processing (NLP): Using the spaCy library for Named Entity Recognition (NER) and rule-based sentiment analysis on Amazon review data.

Project Accomplishments

This project successfully implemented and demonstrated proficiency across three core domains of Machine Learning:

Classical Model Deployment: Trained, evaluated, and visualized a Decision Tree Classifier, showcasing knowledge of model introspection and classical supervised learning workflows.

Deep Learning Architecture: Designed and trained a foundational CNN architecture for image classification on the MNIST dataset, validating understanding of neural networks, convolution, and pooling layers.

Applied NLP: Utilized production-ready libraries (spaCy) to perform practical NLP tasks, specifically Named Entity Recognition and the development of a rule-based sentiment analysis function on simulated review data.

🛠️ Setup and Installation (VS Code on Ubuntu)

Follow these steps to set up your environment and install all necessary dependencies.

1. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment (.venv) to isolate project dependencies.

# 1. Create the virtual environment
python3 -m venv .venv

# 2. Activate the environment (Crucial for Ubuntu/Linux)
source .venv/bin/activate


2. Install Dependencies

With the virtual environment activated, install the required libraries using pip.

# Install core libraries (NumPy, Matplotlib, Pandas)
pip install numpy matplotlib pandas

# Install Scikit-learn (Task 1)
pip install scikit-learn

# Install TensorFlow/Keras (Task 2)
pip install tensorflow

# Install spaCy for NLP (Task 3)
pip install spacy

# Download the small English model for spaCy
python -m spacy download en_core_web_sm


Note: Ensure your VS Code interpreter is set to the Python executable inside the .venv folder.

💻 Running the Code

The project code is structured to run seamlessly within a single Jupyter Notebook (.ipynb).

Open VS Code and ensure the .venv is selected as the Jupyter Kernel.

Open the file containing the project code (e.g., ml_tasks.ipynb).

Run the cells sequentially.

Key Details for Each Task

1. Classical ML: Decision Tree on Iris

Data Source: The Iris dataset is loaded directly using sklearn.datasets.load_iris(). No manual download is required.

Output: The code prints the model's accuracy, the classification report, and generates a visual plot of the Decision Tree structure using matplotlib.

2. Deep Learning: CNN on MNIST

Data Source: The MNIST dataset is loaded directly using tensorflow.keras.datasets.mnist.load_data(). The dataset will be downloaded automatically the first time this function is called.

Model: A simple Convolutional Neural Network (CNN) is defined using Conv2D, MaxPooling2D, and Dense layers.

Output: The script shows the model training progress, evaluation metrics (loss and accuracy), and a visualization of sample predictions.

3. NLP: Named Entity Recognition and Sentiment

Data Source: This task uses a local array of sample strings to simulate Amazon product reviews.

NLP Tools: Utilizes the en_core_web_sm model from spaCy.

Functions:

Named Entity Recognition (NER): Identifies entities like PRODUCT, ORG, or PERSON in the text.

Rule-Based Sentiment: A simple function that calculates sentiment based on the count of nouns, adjectives, and verbs found in the review text.

