Turkish Author Identification Project

This natural language processing project is developed for COME 448 to identify the authorship of Turkish literary text snippets. The dataset includes short text samples from 6 classical Turkish authors: Omer Seyfettin, Sabahattin Ali, Huseyin Rahmi Gurpinar, Halit Ziya Usakligil, Resat Nuri Guntekin, and Ahmet Rasim.

To increase the size of the dataset and make the models more robust, the script applies a data augmentation technique based on random sentence shuffling. The project implements and compares two different approaches: a baseline Logistic Regression model using TF-IDF features, and an advanced deep learning model using a fine-tuned Turkish BERT (dbmdz/bert-base-turkish-cased) architecture.

Running the script automatically prepares the data, applies augmentation, trains both models, and evaluates their performance. It outputs detailed classification reports, visualizes the results with an accuracy comparison chart and a confusion matrix, and runs a custom inference function to test new sentences written in the style of these authors.

Prerequisites
Before running the script, make sure to install the required Python libraries: torch, transformers, scikit-learn, pandas, matplotlib, and seaborn.
