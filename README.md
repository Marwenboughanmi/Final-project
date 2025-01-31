# Final-project

## Overview
This project aims to classify lung images using Convolutional Neural Networks (CNNs). The dataset includes images of various sizes and corresponding labels for different lung conditions such as COVID-19, Viral Pneumonia, and Lung Opacity.

## Directory Structure
The project directory is structured as follows:

## File Descriptions
- `images/`: Contains preprocessed image files in pickle format.
  - `img_preprocessed32.pkl`: Contains 32x32 images.
  - `img_preprocessed64.pkl`: Contains 64x64 images.
  - `img_preprocessed128.pkl`: Contains 128x128 images.
  - `img_preprocessed299.pkl`: Contains 299x299 images.
- `covid_image_data.csv`, `normal_image_data.csv`, `opacity_image_data.csv`, `pneumonia_image_data.csv`: CSV files containing metadata for each category.
- `COVID.metadata.xlsx`, `Lung_Opacity.metadata.xlsx`, `Normal.metadata.xlsx`, `Viral Pneumonia.metadata.xlsx`: Excel files containing detailed metadata for each category.
- `merged_image_data.csv`: CSV file containing merged metadata from all categories.
- `branching_steps.ipynb`: Jupyter notebook for branching steps.
- `CNN_Lungs.ipynb`: Main Jupyter notebook for CNN model.
- `Lungs_CNN_Amar_32.ipynb`, `Lungs_CNN_Amar_64.ipynb`, `Lungs_CNN_Amar_128.ipynb`, `Lungs_CNN_Amar_299.ipynb`: Jupyter notebooks for training the model on different image sizes.
- `Preprocessing.ipynb`: Jupyter notebook for data preprocessing.
- `README.md`: This README file.
- `README.md.txt`: Original README file with additional information.

## Notebooks
There are four main Jupyter notebooks, each corresponding to a different image size:

- `Lungs_CNN_Amar_32.ipynb`: Processes and trains the model on 32x32 images.
- `Lungs_CNN_Amar_64.ipynb`: Processes and trains the model on 64x64 images.
- `Lungs_CNN_Amar_128.ipynb`: Processes and trains the model on 128x128 images.
- `Lungs_CNN_Amar_299.ipynb`: Processes and trains the model on 299x299 images.

## Data
The images are stored in the `images` folder in the following pickle files:

- `img_preprocessed32.pkl`: Contains 32x32 images.
- `img_preprocessed64.pkl`: Contains 64x64 images.
- `img_preprocessed128.pkl`: Contains 128x128 images.
- `img_preprocessed299.pkl`: Contains 299x299 images.

## Usage
1. Clone the repository.
2. Ensure you have the necessary dependencies installed.
3. Run the preprocessing notebook `Preprocessing.ipynb` to prepare the data.
4. Open and run the desired notebook (`Lungs_CNN_Amar_32.ipynb`, `Lungs_CNN_Amar_64.ipynb`, `Lungs_CNN_Amar_128.ipynb`, or `Lungs_CNN_Amar_299.ipynb`) to train the model on the corresponding image size.

## Dependencies
Make sure you have the following dependencies installed:
- Keras
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

You can install the dependencies using pip:
```sh
pip install tensorflow keras numpy pandas matplotlib scikit-learnmodel.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

## Model Compilation
The model is compiled using the Adam optimizer and categorical crossentropy loss function. Here is an example of the model compilation code:

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
## Dataset Source and Mandatory References
The dataset used in this project is sourced from various publicly available repositories. When using this dataset, please ensure to cite the following sources:

- [BIMCV COVID-19](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711)
- [COVID-19 Image Repository](https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png)
- [SIRM COVID-19 Database](https://sirm.org/category/senza-categoria/covid-19/)
- [EuroRad](https://eurorad.org)
- [COVID-19 Chest X-ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- [COVID-19 Chest X-ray Image Repository](https://figshare.com/articles/COVID-19_Chest_X-Ray_Image_Repository/12580328)
- [COVID-CXNet](https://github.com/armiro/COVID-CXNet)
- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
- [Chest X-ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
```
