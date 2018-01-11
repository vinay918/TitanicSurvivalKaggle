# TitanicSurvivalKaggle (Accuracy: 79%, 40th Percentile of Submissions)

This repository contains all my files that I used in my first complete Data Science project. The code is well documented so specific answers about blocks of code can be found within the inline comments. 

In this project I went through the whole process of exploratory data visualisation and analysis, feature engineering, machine learning model creation, and finally prediction on the given test data. 

My goal as I undertook this project was not to get extremely accurate predictions (which doesn't say much as there are many rigged entries on Kaggle), but to learn the whole data science process with a hands-on activity. In addition to that, I wanted to use this challenge as a means to improve my Python skills. 

On my Kaggle profile (https://www.kaggle.com/vinay918), you might notice that I submitted my predictions more than 10 times. The reason for that goes back to the goals I wished to achieve through this challenge. I tried many different steps iteratively to see what improved my score and what didn't, which has given me an idea of how to proceed with my data in the future. 

In the end, my prediction scored 79% and put me in the 40th Percentile of all submissions (which probably includes a ton of rigged submissions). I could go back into my data and change various things such as imputing the ages of individuals using their titles and the median age for individuals with the same titles (Which I'm 79% sure will improve my score as around 300 values for age were missing from the initial dataset, leaving the biggest room for improvement in this feature), or I could use k-fold cross validation instead of my current 80%/20% train/test split. But I feel that I have accomplished my goals in this challenge and I will not spend additional unnecessary time to improve my accuracy by a couple of percent. 

Lessons Learned:

1. Categorical values should be dealt with carefully as they could have some ordinal relationship with the label. (eg: 1 is more likely to produce X than 2 or 3).
2. Imputing features that have a large number of missing data is not the best idea.
3. Make sure your columns are in the right order before feeding into the models while predicting on your test data.
4. Just because the models perform well on your data accuracy wise, it doesn't mean the model is great because precision and recall are equally important. My models were consistently giving me accuracies of around 80%-85% but my Kaggle submissions were 5%-10% lower. When I dug deeper into my model evaluation, I realized that my recall values were pretty low (around 70%), meaning that my models were likely to predict '0' more often, but there must be more '1' solutions on the test data. 
5. The value of Pandas, Matplotlib, and Scikit-Learn is amazing.
6. Utilize Jupyter Notebook environment for clearer workflow and documentation.
