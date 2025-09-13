**L'Oréal X Monash Datathon 2025: TrendSpotter**
This repository contains the project submission for the L'Oréal X Monash Datathon 2025 under the "TrendSpotter" theme. The project aims to use machine learning to identify, analyze, and predict emerging trends from social media comment data.

Members of the Datathon:
1. Chan Zheng Shao (Group Leader)
2. Vicky Leow Ming Fong (Member)
3. Hong Jing Jay (Member)

**Project Files**
comment_cleaned.ipynb: This Jupyter Notebook is used to download the initial CSV file and perform the data cleaning and preprocessing steps using libraries like nltk.

bertopic_model_safetensors: This file contains the pre-trained BERT model used for topic modeling.

bertopic_processed_data.parquet: This file is a Parquet file for efficiently storing the large, processed dataset.

embedded.db: This is the SQLite database file containing the processed data from comment4.csv, including text and numerical embeddings. Due to its size, it is hosted on Google Drive and can be accessed via this link (https://drive.google.com/uc?id=1Wuq8GjDAXPRf0MELPROHPBaoNhTlC0Hb)

filter_google.py: A Python script that leverages GPU acceleration with cuDF, CuPy, and PyTorch to clean text data, generate numerical embeddings using a pre-trained model, store the data in an SQLite database, and upload it to Google Drive.

modelling+clustering+temporal.ipynb: This Jupyter Notebook is the core of the analysis pipeline. It is responsible for modeling, clustering, and performing temporal analysis using the data stored in embedded.db.

Final Sentiment Analysis.ipynb: Applies TF-IDF-based topic modeling and VADER sentiment analysis to identify key themes and their associated sentiments, finally visualizing these results through bar charts and word clouds.

**Project Setup and Running the Project**
To set up and run this project, follow the steps below:

1. Clone the Repository
git clone [https://github.com/jingjayy/monashxloreal.git]
2. Download the Dataset
The embedded.db database file is crucial for the analysis. Please download it from the Google Drive link provided above and place it in the project's root directory.
3. Install Dependencies
Install the required Python libraries.
4. Run the Pipeline

**The project can be run by executing the Jupyter Notebooks in the specified order (make sure to download the dataset)**
filter_google.py -> modelling+clustering+temporal.ipynb -> Final Sentiment Analysis.ipynb

**Identified Limitations**
1. Scalability & Infrastructure
 Although we leveraged a powerful platform like Vast.ai to run our models for speed and efficiency, this approach is not a permanent solution for a production environment due to the associated costs, and a future plan would involve migrating to a cloud-based infrastructure.
2. Data and Model Pipeline
 Migrate our local-based prototype to a scalable cloud-based infrastructure to handle real-time data and the large volume of comments required for L'Oréal's global business needs.
3. Resource Allocation
 Expansion would require additional time and budget for data access APIs from social media platforms and for the computational resources needed to run and fine-tune large multilingual models in a cloud environment.