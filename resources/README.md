# Fashion-MNIST: Multi-Model Classification and Personalized Recommendations

This repository showcases a comprehensive machine learning workflow using **Fashion-MNIST**, a well-known dataset for image classification. Multiple models (SVM, KNN, CNN) are evaluated and compared, and a "dummy user profile" scenario demonstrates how to leverage the best-performing model to provide personalized fashion recommendations.

---

## Table of Contents
1. [Overview](#overview)  
2. [Notebook Workflow](#notebook-workflow)  
3. [Key Features](#key-features)  
4. [Dataset](#dataset)  
5. [Implementation Details](#implementation-details)  
6. [Results & Visualizations](#results--visualizations)  
7. [Personalized Recommendations](#personalized-recommendations)  
8. [Next Steps](#next-steps)  
9. [Running the Project](#running-the-project)  
   - [Install Python 3.8+](#install-python-38)  
   - [(Optional) Create a Virtual Environment](#optional-create-a-virtual-environment)  
   - [Install Dependencies](#install-dependencies)  
   - [Execution](#execution)  
   - [Data Files](#data-files)  
10. [License](#license)  
11. [Contact](#contact)

---

## Overview
This project explores three classification approaches on **Fashion-MNIST**:
- **SVM** (Support Vector Machine)  
- **KNN** (K-Nearest Neighbors)  
- **CNN** (Convolutional Neural Network with Keras/TensorFlow)

Key steps include:
- **Hyperparameter tuning** with `GridSearchCV` for SVM and KNN, plus a manual search for CNN.
- **Comparative evaluation** of models (accuracy, precision, recall, F1-score, confusion matrices).
- **A dummy user profile** to illustrate how the best model can feed into a recommendation-like system.

---

## Notebook Workflow
1. **Data Loading**  
   Loads Fashion-MNIST subsets (5,000 training samples, 1,000 testing samples) for demonstration.

2. **Exploratory Data Analysis**  
   Displays random images and their classes (e.g., T-shirt/top, Trouser, Pullover).

3. **Preprocessing**  
   Normalizes pixel values, reshapes data (28×28×1 for CNN), and one-hot encodes labels.

4. **Hyperparameter Tuning**  
   - **SVM**: Searches over `C`, `kernel`, `gamma` using `GridSearchCV`.  
   - **KNN**: Searches over `n_neighbors`, `weights`, and distance metrics with `GridSearchCV`.  
   - **CNN**: Tests different optimizers (`adam`, `sgd`) and activations (`relu`, `sigmoid`).

5. **Model Evaluation**  
   Prints accuracy, classification reports, and confusion matrices.

6. **Model Comparison**  
   Compares SVM, KNN, and CNN side-by-side (Accuracy, Precision, Recall, F1) and visualizes epoch/batch size effects on CNN.

7. **Personalized Recommendations**  
   Generates a dummy user profile (random class and color preferences).  
   Uses the highest-accuracy model for final predictions, illustrating how to recommend items matching the user’s top classes.

---

## Key Features
- **Multi-Model Benchmark**: Compares SVM, KNN, and CNN.  
- **Hyperparameter Optimization**: Demonstrates systematic searches for SVM, KNN, and CNN.  
- **Visual Insights**: Includes confusion matrices, performance plots, spider charts, and bar charts.  
- **Dummy User Profile**: Simulates basic recommendation logic.

---

## Dataset
**Fashion-MNIST**:
- 28×28 grayscale images with 10 classes (e.g., T-shirt/top, Trouser, Pullover, etc.).
- Original dataset: 60,000 training images, 10,000 test images.
- This project uses partial data (5,000 training samples, 1,000 testing samples) for faster demonstration.

For more details, visit [Fashion-MNIST on GitHub](https://github.com/zalandoresearch/fashion-mnist).

---

## Implementation Details
- **SVM**  
  Tuned with `GridSearchCV` (various `C`, `kernel`, `gamma`). Trained on flattened, normalized vectors (784 dimensions).

- **KNN**  
  Tuned with `GridSearchCV` over `n_neighbors`, `weights`, and distance metrics. Trained on normalized vectors.

- **CNN**  
  Built with multiple convolutional layers, pooling, dropout, and dense layers (Keras/TensorFlow).  
  Tests a small search space of optimizers (`adam`, `sgd`) and activations (`relu`, `sigmoid`).  
  Evaluates each combination to find the best settings.

- **Performance Metrics**  
  Accuracy, Precision, Recall, F1-score from `classification_report`.  
  Confusion matrix visualized with `seaborn.heatmap`.

---

## Results & Visualizations
- **Comparative Plot**: Displays Accuracy, Precision, Recall, and F1 for each model.  
- **CNN Confusion Matrix**: Shows class-wise performance for the CNN.  
- **Spider Chart & Bar Chart**: Visualizes dummy user’s category and color preferences.  
- **Recommendation Samples**: Highlights items in the user’s top-preferred classes.

---

## Personalized Recommendations
1. **Dummy User Profile**  
   - Randomly generated labels (0–9) and color preferences (Red, Blue, Green, etc.).  
   - Summarized via spider (radar) chart and bar chart.

2. **Integration**  
   - Picks the best model by accuracy (SVM, KNN, or CNN).  
   - Classifies new items, displays suggestions matching user’s top categories.

---

## Next Steps
- **Use Full Dataset**: Train on all 60k images for higher accuracy.  
- **Expand Hyperparameter Tuning**: Explore additional SVM kernels, deeper CNNs, or advanced optimizers.  
- **Refine Recommendation Logic**: Incorporate real user data, more robust preference modeling.

---

## Running the Project

### Install Python 3.8+
- **Windows**: [Download from python.org](https://www.python.org/downloads/windows/) and check “Add Python to PATH” during installation.  
- **Linux (Debian/Ubuntu)**:
  
      sudo apt-get update
      sudo apt-get install python3 python3-pip
  
- **macOS**:
  
      brew install python3
  
  Or use the official [python.org installer](https://www.python.org/downloads/mac-osx/).

### (Optional) Create a Virtual Environment

    cd <YourRepositoryName>
    python -m venv env
    # Activate the virtual environment:
    # On Windows:
    env\Scripts\activate
    # On macOS/Linux:
    source env/bin/activate

### Install Dependencies
No `requirements.txt` is included. Install the needed libraries individually:

    pip install numpy
    pip install pandas
    pip install matplotlib
    pip install seaborn
    pip install scikit-learn
    pip install tensorflow
    pip install scikeras

### Execution
1. **Launch Jupyter Notebook**

       jupyter notebook

2. **Open the Notebook**  
   - Open **FashionMNIST_ModelComparison_and_Recommendations.ipynb** (or your chosen name).  
   - Run the cells sequentially (data loading, EDA, model training, evaluation, recommendation steps).

### Data Files
- Ensure `fashion-mnist_train.csv` and `fashion-mnist_test.csv` are in the correct folder.  
- Update file paths in the notebook if necessary.

---

## License
You are free to use, modify, and distribute this project under the terms of the **MIT License**.  
The MIT License is a short, permissive open-source license that grants users significant freedom while minimizing legal complexity.  

---

## Contact
**Name**: Ashish Paka  
**Email**: [ashishpaka1998@gmail.com](mailto:ashishpaka1998@gmail.com)  
**LinkedIn**: [ashish-probotics](https://www.linkedin.com/in/ashish-probotics)

Feel free to reach out for any questions, suggestions, or collaboration inquiries!
