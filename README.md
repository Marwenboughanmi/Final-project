# Final-project

## Overview
This project aims to classify lung images using Convolutional Neural Networks (CNNs). The dataset includes images of various sizes and corresponding labels for different lung conditions such as COVID-19, Viral Pneumonia, and Lung Opacity. 

## Google Drive
Google Drive has the majority of the data of the project, the link is https://drive.google.com/drive/folders/1ZcW3BefqcWtmretHrRnnih8GQxFUadbP?usp=drive_link 

## Directory Structure
The project directory is structured as follows:

```
├── `README.md`                     # Project overview and instructions
├── `XRay-CSVs/`                    # CSV and Excel files containing metadata and image data
│   ├── `COVID.metadata.xlsx`       # Metadata for COVID-19 images
│   ├── `Lung_Opacity.metadata.xlsx`# Metadata for Lung Opacity images
│   ├── `Normal.metadata.xlsx`      # Metadata for Normal images
│   ├── `Viral Pneumonia.metadata.xlsx` # Metadata for Viral Pneumonia images
│   ├── `covid_image_data.csv`      # CSV file with COVID-19 image data
│   ├── `merged_image_data.csv`     # CSV file with merged image data
│   ├── `normal_image_data.csv`     # CSV file with Normal image data
│   ├── `opacity_image_data.csv`    # CSV file with Lung Opacity image data
│   └── `pneumonia_image_data.csv`  # CSV file with Viral Pneumonia image data
├── `X_Ray_Analysis_Final.pptx`     # Final presentation of X-ray analysis
├── `X_Ray_Images/`                 # Directory containing X-ray images
│   ├── `COVID/`                    # Folder with COVID-19 images
│   ├── `Lung_Opacity/`             # Folder with Lung Opacity images
│   ├── `Normal/`                   # Folder with Normal images
│   └── `Viral_Pneumonia/`          # Folder with Viral Pneumonia images
├── `archive/`                      # Archive folder for backup and old files
├── `notebooks/`                    # Jupyter notebooks for model training and analysis
│   ├── `CNN_Branching_Model_32.ipynb` # Notebook for CNN branching model with 32x32 images
│   ├── `CNN_Model_Classification_32.ipynb` # Notebook for CNN classification model with 32x32 images
│   ├── `CNN_Model_Classification_64.ipynb` # Notebook for CNN classification model with 64x64 images
│   ├── `CNN_Model_Classification_128.ipynb` # Notebook for CNN classification model with 128x128 images
│   ├── `Gradio.ipynb`              # Notebook for Gradio interface
│   └── `Preprocessing.ipynb`       # Notebook for data preprocessing
├── `pkl_files/`                    # Directory for preprocessed image files in pickle format
│   └── `img_preprocessed32.pkl`    # Preprocessed images with 32x32 resolution
```

## Usage
1. Clone the repository.
2. Ensure you have the necessary dependencies installed.
3. Run the preprocessing notebook `Preprocessing.ipynb` to prepare the data.
4. Open and run the desired notebook to train the model on the corresponding image size.

## Dependencies
Make sure you have the following dependencies installed:
- Keras
- TensorFlow2.17.1
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
The dataset used in this project is sourced from various publicly available repositories. 

References:
-M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.
-Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. arXiv preprint arXiv:2012.02238.


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
Dataset used in this project is licensed under Creative Commons License. Refer to the dataset site for additional information: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

The project is licensed under the MIT license.
```
