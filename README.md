# Detecting-Mental-Instability-in-English-and-Bengali-Text-A-Machine-Learning-Approach


Mental health is a critical concern, and social media platforms have gained popularity for discussing such issues, especially among young adults. This study employs machine learning, deep learning, and transformer-based learning techniques to identify early-stage mental instability in texts written in Bengali and English. The datasets were constructed using English data collected from Reddit. Due to a lack of resources, the acquired data for Bengali data was manually reviewed and sanitized after being translated using the Google API into Bengali. The resulting datasets contain 10,710 English and 10,287 Bengali text data, respectively, categorized into ten mental health classes and one neutral class. Cohen's Kappa scores for the English and Bengali datasets were 91\% and 89\%, respectively. The multilingual model XLM-RoBERTa achieved the highest accuracy of 80.53\% for English and 76.68\% for Bengali. LIME was utilized to explain the model's output. This research investigates the potential use of social media data to identify mental instability in Bengali and English, significantly contributing to creating valuable datasets for mental health detection. The findings demonstrate the promise of natural language processing (NLP) techniques in enhancing mental health detection and intervention, particularly in low-resource languages.
Future research could involve utilizing social media data to detect other mental abnormalities in different languages, building on the insights obtained from this study. The study's approach of gathering data from social media and utilizing LIME has potential implications for improving mental health interventions and treatments. 

## Dataset

Data is collected from user posts on Reddit using web scraping. There are support groups on Reddit with people wanting to know about and suffering from mental health issues, and experts in those groups give them consultancy. These groups are known as "subreddits." We collected data from each subreddits for every mental state we worked on. There are subreddits specialized in mental health issues such as Bipolar, Schizophrenia, Addiction, Alcoholism, Asperger’s, Neutral, Suicidal Thought, Anxiety, Depression, and Self Harm. All these mental health issues. Web scraping techniques are used in this process. We used Reddit's API to scrape data from Reddit. Using scraping, we scraped 12000 English data from Reddit.

![Data-Collect-(Small)](https://user-images.githubusercontent.com/74653056/236351754-7ea88bd7-b558-4336-a474-8e254f7224e4.png)

<p align="center">
Fig: Data Collection Process
</p>


| **Class Name** | **English Data** | **Bengali Data**
| :-------- | :------- | :------- 
| Anxiety | 1024 | 991
| Bipolar | 1024 | 1000
| Borderline Personality | 1024 | 995
| Depression | 1024 | 943
| Schizophrenia | 1024 | 988
| Suicidal Thought | 1024 | 1001
| Alcoholism | 999 | 986
| Addiction | 998 | 921
| Asperger’s | 799 | 789
| Self Harm | 746 | 671
| Neutral | 1024 | 1002


## Embedding

- Word vector/embedding methods like `TF-IDF`, `Word2Vec`, `GloVe` and `fastText` was used 
- Context embedding methods like `BERT` and `RoBERTa` was used. 


## Model Accuracy

From the dataset 80% data was used for training and 20% for testing.


| **Model** | **Embedding** | **Accuracy (English)** | **Accuracy (Bangla)**
| :-------- | :------- | :-------- | :------- 
| Naïve Bayes | TF-IDF | 63.12% | 51.51%
| SVM | TF-IDF | 70.82% | 57.58%
| XGBoost | TF-IDF | 71.57% | 58.07%
| AdaBoost | TF-IDF | 62.89% | 51.55%
| Random Forest | TF-IDF | 70.26% | 55.25%
| Stochastic Gradient Descent | TF-IDF | 63.02% | 58.31%
| Logistic Regression | TF-IDF | 71.34% | 58.79%
| CNN | Word2Vec | 60.13% | 55.34%
| CNN | GloVe | 61.95% | 56.51%
| CNN | fastText | 60.46% | 59.69%
| BiLSTM | Word2Vec | 63.63% | 59.69%
| BiLSTM | GloVe | 63.63% | 59.96%
| BiLSTM | fastText | 64.33| 60.79%
| CNN+BiLSTM | Word2Vec | 63.07% | 58.06%
| CNN+BiLSTM | GloVe | 63.87% | 58.21%
| CNN+BiLSTM | fastText | 63.83% | 58.06%
| BERT Base | BERT | 80.35% | -
| Bangla BERT | BERT | - | 72.55%
| Multilingual BERT | BERT | 78.90% | 75.36%
| XLM-RoBERTa | RoBERTa | 80.53% | 76.68%


## Explainable AI

#### Explainer function from the Ktrain was used to explain model's output using `LIME`. 



