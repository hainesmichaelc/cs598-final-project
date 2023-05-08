# CS598 Deep Learning for Healthcare - Final Project Submission

*Author*: Michael Haines | mhaines2@illinois.edu

### Citation to Original Paper

This notebook is an attempt to recreate a model similar to CEHR-GAN-BERT from a paper titled ["Few-Shot Learning with Semi-Supervised Transformers for
Electronic Health Records"](https://static1.squarespace.com/static/59d5ac1780bd5ef9c396eda6/t/62e97a83db963576550e863e/1659468419684/github.com/healthylaife/CEHR-GAN-BERT).

### Runtime Dependencies

The runtime dependencies can be found in the `requirements.txt` file, and can be installed using `pip install -r requirements.txt`.

### Data Download Instructions

The code uses the ICU module of the [MIMIC-IV dataset](https://physionet.org/content/mimiciv/2.2/). This dataset is protected, and you must request access from PhysioNet before downloading. After downloading the data, upload the data into your personal Google Drive in a `./data/mimiciv/` directory. When properly configured, the ICU module should be available from the `./data/mimiciv/icu/` sub-directory of your Google Drive root directory.

### Pre-processing Notebook

The `Pre-processing.ipynb` notebook contains the end-to-end pre-processing required to produce the tokenized datasets. This creates more detailed cohort data, visualizations, and summary statistics along the way. Note that only the tokenized datasets, vocabularies, and charts are published to source control; this notebook will create:

* Pre-processed cohort of patient visits used for model training in `./data/cohort/` directory
* Summary statistics for features in `./data/summary/` directory

### Model Training Notebook

The `ModelTraining.ipynb` notebook trains all BERT-based models used in the analysis. Given that this code requires a GPU to run efficiently, and still takes several hours to complete, I have comitted the results of my experiments to source control in the `./saved_models/` directory. Please use Google Colab if you wish to re-train the models.

### Evaluation Notebook

The `Evaluation.ipynb` notebook takes the results of the models that I trained and performs the evaluations.

### BONUS - Descriptive Notebook

I am turning in the evaluation notebook as my descriptive notebook. However, if that one is not descriptive enough, further description can be found in the `Pre-processing.ipynb` and `ModelTraining.ipynb` notebooks.

### References

To get started, I leaned on the following code bases:

* [MIMIC-IV Data Pipeline project](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline) to pre-process the MIMIC-IV dataset
* [Original study's code base](https://github.com/healthylaife/CEHR-GAN-BERT)

Thank you to the original authors for providing a great starting point for the analysis. 