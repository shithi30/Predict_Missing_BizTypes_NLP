*TallyKhata* platforms's 60% merchants would voluntarily fill in their business types (grocery, pharmacy, electronics, fashion house etc), and for identifying the rest:

- I train *Word2Vec* NLP models using existing data - journal descriptions, shop names and supplier names.
- After training, I poll among inferences from all 3 models, label missing business types and deploy via *Streamlit*.

**Tech Stack:** `Python` `TensorFlow` `Keras` `PostgreSQL` `h5` `Streamlit`

### A. Implementation Using Journal Descriptions

**Step-01:** Preparation of the Dataset

<p align="center"><img width="440" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/03933441-676c-4bbb-967a-e284d75a999b"></p>

- **a.** First, we replace all Bengali digits by their English counterparts.
- **b.** Second, we ensure the strings contain all Bengali words (in version-01) and no punctuation marks. 
- **c.** Third, we remove unnecessary whitespaces from the training strings. 
- **d.** Fourth, we aggregate the last 7 day's available descriptions against available merchants. 
- **e.** Finally, we randomly shuffle the dataset for a representative distribution in each sample.

**Step-02:** Fitting the *Word2Vec* Model

- **a.** Use IDE: Google Colab and mount to Google Drive, for fetching training data

  ```Python
  # import
  import io
  import csv
  import tensorflow as tf
  import numpy as np
  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  import matplotlib.pyplot as plt
  # mount
  from google.colab import drive
  drive.mount('/content/drive')
  ```
  
- **b.** Set hyperparameters for *Word2Vec* model

<p align="center"><img width="220" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/3ccb6476-bf74-4b23-ba18-452eadc572d3"></p>

- **c.** Split full dataset into training and dev sets

<p align="center"><img width="470" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/f4a66c29-2d08-44b6-b871-28a149d7aefe"></p>

- **d.** Define the model 
<p align="center"><img width="400" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/926f2e3f-e558-44a9-9158-7c3a9ddb56eb"></p>

- **e.** Train the model

<p align="center"><img width="660" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/eabdade3-70df-4c1e-8b72-c7645cc49cd5"></p>

**Step-03:** Generate/Inspect Predictions 

- **a.** Generate output as per the highest probable class

<p align="center"><img width="462" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/21e0f6d9-5c84-44b3-98ba-5076f0972b66"></p>

- **b.** Inspect outputs

<p align="center"><img width="700" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/c4014b62-7ea9-422f-be64-90f386af6821"></p>

### B. Implementation Using Supplier Names

**Step-01:** Preparation of the Dataset

<p align="center"><img width="463" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/de4772de-518b-431b-9a17-dbbd12673452"></p>

All steps are the same as before, except that: 
- **a.** We consider both Bengali and English names.
- **b.** For English names, we convert all the letters to lowercase for the sake of cleanliness of data. 
- **c.** Additionally, we classify on 7 classes and remove separators (e.g., _ ) from class names. 

**Step-02:** Fitting the *Word2Vec* Model 

**Steps - a, b, c, d** are as before. 

- **e.** Train the Model 

<p align="center"><img width="650" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/379f80f4-9665-412c-8a08-96216c3e0292"></p>

**Step-03:** Generate/Inspect Predictions

- **a.** Generate output as per the highest probable class
- **b.** Inspect outputs  

<p align="center"><img width="600" alt="c4" src="https://github.com/shithi30/Prediction_of_Missing_BizTypes_NLP/assets/43873081/14e5d7c2-a0f6-4564-8f0c-4f277d61a0db"></p>

The models identified 40% (missing) business types. The Growth & Marketing team thus gained the capability to custom-message these merchants as per the needs of their respective businesses. 

*<strong>Note:</strong> Not all data/models/parameters are revealed, due to org's governance policy.*




