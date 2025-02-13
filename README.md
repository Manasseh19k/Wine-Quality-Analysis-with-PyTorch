# Wine Quality Analysis with PyTorch

This Python script demonstrates how to load and analyze wine quality data from a CSV file using **PyTorch**, **NumPy**, and other libraries. It covers fundamental PyTorch operations such as:

- Converting NumPy arrays to `torch.Tensor` objects  
- Indexing and slicing with boolean masks (advanced indexing)  
- Basic data normalization and summarization  
- Simple threshold-based classification

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features and Steps](#features-and-steps)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Code Highlights](#code-highlights)
- [License](#license)
- [Contact](#contact)

---

## Overview

1. **Load** the wine quality CSV data (white wine) from a specified path into a NumPy array.  
2. **Convert** the data into PyTorch tensors for efficient numerical computations.  
3. **Explore** data distributions by calculating means and variances for different quality categories.  
4. **Apply** boolean masks and advanced indexing to create subsets (e.g., “bad,” “mid,” “good” wines).  
5. **Perform** a simple threshold-based classification and compare predictions to actual labels.  
6. **Review** classification performance metrics (e.g., how many predictions matched the actual labels).

This approach provides an illustrative example of how PyTorch tensors can be used for data preprocessing and basic classification tasks, even outside of building full neural networks.

---

## Dataset

- The script expects the file **`winequality-white.csv`**.  
- By default, it uses:
  ```text
  Deep_Learning\Deep Learning with pytorch\dlwpt-code-master\data\p1ch4\tabular-wine\winequality-white.csv
  ```
- The first row contains column headers (like "fixed acidity", "volatile acidity", etc.).  
- Each subsequent row corresponds to a single wine sample’s measurements.  
- The last column contains the quality rating on a scale (e.g., 3–9).

**Note**: You may replace this path with your own local path. Make sure the CSV structure matches the script’s expectations.

---

## Features and Steps

1. **Loading CSV Data**:  
   - Utilizes Python’s built-in `csv` module to read header information.  
   - Uses NumPy’s `np.loadtxt` for numerical data loading and conversion.

2. **Tensor Conversion**:  
   - Converts the NumPy array into a PyTorch tensor using `torch.from_numpy()`.  

3. **Splitting Features and Labels**:  
   - Slices the last column as the target (wine quality).  
   - Slices the rest as input features (e.g., acidity, sulphates, etc.).

4. **One-Hot Encoding**:  
   - Demonstrates creating a one-hot encoded tensor (`target_onehot`) for labels.

5. **Data Normalization**:  
   - Calculates mean and variance per column (feature) and standardizes the data.

6. **Boolean Masking & Indexing**:  
   - Shows how to filter data by target categories (e.g., `target <= 3` for “bad” wines).  
   - Summarizes subsets via mean values.

7. **Threshold-Based Classification**:  
   - Applies a threshold (`total_sulfur_threshold`) to classify wines as “good” or “not good.”  
   - Compares predictions to the actual labels and calculates the match rate.

---

## Dependencies

Below is a minimal set of libraries required. A sample **`requirements.txt`** might look like:

```txt
numpy>=1.19.5
torch>=1.9.0
torchtext>=0.10.0
matplotlib>=3.3.2
```

Although the script imports `json` and `torchtext`, you might not strictly need them if you’re not using their functionality. If you do, make sure they’re included in your environment.

---

## Installation

1. **Clone the Repository** (or copy the script):
   ```bash
   git clone https://github.com/YourUsername/YourRepository.git
   cd YourRepository
   ```

2. **(Optional) Create a Virtual Environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install numpy torch torchtext matplotlib
   ```
   Or, if you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Adjust the File Path** to your CSV if needed:
   ```python
   wine_path = r"C:\Path\to\winequality-white.csv"
   ```

---

## Usage

1. **Open/Run the Script** (e.g., `wine_analysis.py`) in your Python environment:
   ```bash
   python wine_analysis.py
   ```
2. **Observe the Console Output**:  
   - You’ll see printed shapes and data about the mean/variance.  
   - You’ll see classification performance using the threshold approach.

3. **Interpret Results**:  
   - Look for lines showing how many “bad,” “mid,” or “good” wines exist.  
   - Check the final match metrics (`n_matches`, `n_matches / n_predicted`, `n_matches / n_actual`) to gauge how well the threshold approach performed.

---

## Code Highlights

- **Advanced Indexing**:  
  ```python
  bad_data = data[target <= 3]
  mid_data = data[(target > 3) & (target < 7)]
  good_data = data[target >= 7]
  ```
  This filters rows based on boolean conditions and is very efficient in PyTorch.

- **One-Hot Encoding**:  
  ```python
  target_onehot = torch.zeros(target.shape[0], 10)
  target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
  ```
  Uses `scatter_` to set a 1 in the appropriate column for each wine’s quality rating.

- **Threshold Classification**:
  ```python
  predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
  actual_indexes = target > 5
  # Compare predicted vs. actual
  n_matches = torch.sum(actual_indexes & predicted_indexes).item()
  ...
  ```
  A simple example of classification logic and performance checking.

---

## License

You can choose a license (e.g., MIT, Apache 2.0) or remove this section if you prefer. For example:

```
This project is licensed under the MIT License - see the LICENSE file for details.
```

---

## Contact

- **Author**: [Frank Kendemah](https://github.com/YourUsername)
- **Email**: [joseph74790@gmail.com](mailto:joseph74790@gmail.com)

Feel free to reach out for questions, suggestions, or feedback!

---

*Enjoy analyzing wine data and exploring PyTorch operations!*
