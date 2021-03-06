# CrabsMLearning
## Description
This project is developed as a useful machine learning research tool. Currently, is
in alpha development stage, which means, some methods may be unstable or change between
the updates.

## Installation
The project includes a file named **setup.txt**. The file contains the
appropriate commands needed to be installed.

### Dependencies
The program has been developed and tested using the following packages:

- pyside2==5.15.2
- PyQt5==5.15.4
- QDarkStyle==3.0.2
- pandas==1.2.3
- matplotlib==3.4.1
- sklearn==0.24.1
- openpyxl==3.0.7
- tensorflow==2.4.1
- seaborn==0.11.1
- dtw-python==1.1.10
- keras-tuner==1.1.0

# Documentation

## Machine Learning Regression

### Step 0: Open the Application Window
![00_OpenWindow.png](readme/machine_learning_regression/00_OpenWindow.png)

### Step 1: Open the Machine Learning Regression Window
Follow the path to open the Machine Learning Regression Window

**Tools &#8594; Machine Learning &#8594; Machine Learning Regression**

![01_MachineLearningRegression.png](readme/machine_learning_regression/01_MachineLearningRegression.png)

### Step 2: Machine Learning Window
In the Machine Learning Regression Window on the left down corner there is the green cross button. 
Press it to open the add file window.

![02_MachineLearningRegressionWindows.png](readme/machine_learning_regression/02_MachineLearningRegressionWindows.png)

### Step 3: Add File
The application, currently, supports CSV and XLSX data table formats. Follow the path and select
the correct file format and then press open or double click to file for opening the file.

![03_OpenDataset.png](readme/machine_learning_regression/03_OpenDataset.png)

### Step 4: Opened File
After a while, the name of the file will appear in the Opened File List and the
columns contained in the Column List.

![04_TheDatasetOpened.png](readme/machine_learning_regression/04_TheDatasetOpened.png)

### Step 5: Select Machine Learning Input/Output Columns
There are multiple combinations to create a dataset for a machine learning. The application
currently supports:
- Select the columns, which will be used in the creation of **Input Array (X)** and 
  **Output Array (Y)**.
- If the dataset includes some categorized features (e.g. Countries) a column needs th be 
  added in **Primary/Common Event Column (Optional)** cell. This column will be used to create
  an Input Array (X) and an Output Array (Y), for each category and calculate statistical information
  later for each event (e.g. the accuracy of the global model for each country).

![05_CreateInputOutputFiles.png](readme/machine_learning_regression/05_CreateInputOutputFIles.png)

After selecting the columns for each list, the next step is to edit the Machine Learning Setting.

### Step 6: Machine Learning Setting &#8594; General

The General tab, contains the basic information of the Machine Learning process, such us the percentage of
test and validation sets, the path for the output folder (which stores the created model(s) and other statistical
information), different methods for creating the X and Y arrays (sequential for time-series analysis, average for
statistical analysis, sequential-average for time-series statistical analysis), method usage index (indicated 
how many rows will be used to create an input/output row for training-validation, test sets, e.g. for time-series
analysis if method index=7 then a weekly range will be applied), filter usage index (an index used for data convolve)
and multifile training processing (a parameter used for running multiple machine learning processes at the same
time, currently only the Linear Training-Testing is supported).

![06_SetMachineLearningGeneralSettings.png](readme/machine_learning_regression/06_SetMachineLearningGeneralSettings.png)

### Step 7: Machine Learning Setting &#8594; Regression Methods

The Regression Method tab, contains all the available common regressors. Most of the methods are currently under
development.

![07_SetMachineLearningRegressioMethods.png](readme/machine_learning_regression/07_SetMachineLearningRegressioMethods.png)

### Step 8: Machine Learning Setting &#8594; Deep Regression Methods

This tab contains the deep machine learning supported by the application. Different deep machine learning models
can solve, different problems, thus this tab will be enriched and updated accordingly, with models, which solve
different problems.

![08_SetMachineLearningDeepRegressioMethods.png](readme/machine_learning_regression/08_SetMachineLearningDeepRegressioMethods.png)

### Step 9: Machine Learning Setting &#8594; Signal Compare Methods

Signal compare methods tab, contains some statistical methods, which used afterwards. These methods, run after the
training of the model(s) and compare the forecasted values, with the real values. Each method produces significant
statistical metrics, which may help the researcher to improve the accuracy of the models in next executions. Also,
these metrics may help the researcher to find additional correlation(s) of the dataset.

These metrics, are optional. The application calculated by default the most common statistical errors:
mean absolute error (MAE), root mean squared error (RMSE) and max error and export them in a table format.

![09_SetMachineLearningSignalCompareMethods.png](readme/machine_learning_regression/09_SetMachineLearningSignalCompareMethods.png)

### Step 10: Execute
Finally, by pressing the Execute button the application perform the machine learning process. 

![10_Execute.png](readme/machine_learning_regression/10_Execute.png)

# Example Usage
The application has been tested using a COVID-19 dataset downloaded from Our World in Data
(https://github.com/owid/covid-19-data/tree/master/public/data) for the creation of four
deep learning models: (a) Convolutional 1D - Long Short Term Memory; (b) Gated Recurrent Unit;
(c) Long Short Term Memory; and (d) Simple Recurrent Neural Network. The architecture is summarized
bellow:

![01_Architecture.jpg](readme/covid_case_study/01_Architecture.jpg)

Two Global European Union models (case studies) created, using a time series dataset for the period of
March 1st 2020 to September 15th 2021, for each deep learning method, for 12 European countries (Austria, Belgium, 
Denmark, Estonia, Finland, France, Germany, Ireland, Italy, Netherlands, Portugal, and Romania).

- Case Study 1: The model uses as input the mitigation measures (School Closures, Workplace Closures, Cancel Public Events, Restriction in Gatherings, 
Close Public Transport, Stay Home Requirements, Public Information Campaigns, Restrictions Internal Movements, 
International Travel Controls, Facial Coverings) and the New Cases per Million and New Deaths per Million calculated from the
dataset in a weekly period (method index = 7). The output was the New Cases per Million and New Deaths per Million for the next
week.

- Case Study 2: The model uses as input the mitigation measures (School Closures, Workplace Closures, Cancel Public Events, Restriction in Gatherings, 
Close Public Transport, Stay Home Requirements, Public Information Campaigns, Restrictions Internal Movements, 
International Travel Controls, Facial Coverings) and the ICU-Patients per Million and HOSP-Patients per Million calculated from the
dataset in a weekly period (method index = 7). The output was the ICU-Patients per Million and HOSP-Patients per Million for the next
week.

The following figure summarized the results of this research. In the first case study, the results 
showed, that all four methods can be used for the estimation of future cases and deaths, however 
SimpleRNN resulted better than the other methods, for most of the countries. In the second case 
study, LSTM is preferable, because it performed good in both ICU admissions and hospitalized 
patients predictions. 

![00_Results.png](readme/covid_case_study/00_Results.png)

# Licence
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[LICENCE](LICENSE)


# Project State
Alpha Version: Still Under Development. Feel free to mail me for ideas or work with me to set up the best interface for Statistical, Deep/Machine Learning and Big Data Analysis.
