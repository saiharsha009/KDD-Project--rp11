
# Project Group 11

### Team Members

1. Laxmi Harika Bibireddy
2. Sai Harsha Chapala
3. Yashwanth Chowdary Kanaparthi
4. Chaitanya Kurati
5. Vamsidhar Reddy Yarrabelly

## Data Source:

**Dataset Link:** [Students Health and Academic Performance](https://www.kaggle.com/datasets/innocentmfa/students-health-and-academic-performance)

## Project Overview:

This project aims to explore how students' health impacts their academic performance using clustering techniques. By grouping students based on various health and academic features, we can gain insights into patterns and relationships within the data.

## Project Type

- **Type of Project:** Predictive
- **Learning Type:** Supervised learning
- **Methods:** Classification and regression


#### Relevant Domain Information (links to two or more articles that relate to your research question; one will most likely come from the link to the data)
1.https://researchgate.net/publication/377499021_Impact_of_Mobile_Phone_usage_on_School_Students'_Academic_Performance_SSAP_Insights_from_COVID_19 
2.https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9651103/
3.https://www.sciencedirect.com/science/article/pii/S2451958821000622


## Files

- `data.csv`: The dataset with information about students' health and academic performance.
- `kddproject`: The main script for data preprocessing, clustering, and visualization.

## Libraries used

Ensure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install them using pip:


pip install pandas numpy matplotlib seaborn scikit-learn

        
### Data Description

``The dataset includes the following columns:``

Names: Student names

Age: Student ages (in years)

Gender: Male/Female

Mobile phone: Do students own a mobile phone? (Yes/No)

Mobile Operating System: Type of mobile operating system used (e.g., Android, iOS, Other)

Mobile phone use for education: Do students use their mobile phone for educational purposes? (Sometimes/Frequently/Rarely)

Mobile phone activities: List of mobile phone activities used for educational purposes (e.g., online research, educational apps, email, online learning platforms)

Helpful for studying: Do students find mobile phone use helpful for studying? (Yes/No)

Educational Apps: List of educational apps used

Daily usages: Average daily time spent using mobile phone for educational purposes (in hours)

Performance impact: How does mobile phone use impact academic performance? (Agree/Neutral/Strongly agree)

Usage distraction: Does mobile phone use distract from studying? (During Exams/Not Distracting/During Class Lectures/While Studying)

Attention span: Has mobile phone use affected attention span? (Yes/No)

Useful features: What features of mobile phones are useful for learning? (e.g., Internet Access, Camera, Calculator, Notes Taking App)

Health Risks: Are students aware of potential health risks associated with excessive mobile phone use? (Yes/No/Only Partially)

Beneficial subject: Which subjects benefit most from mobile phone use? (e.g., Accounting, Browsing Material, Research)

Usage symptoms: Are students experiencing any physical or mental symptoms related to mobile phone use? (e.g., Sleep disturbance, headaches, Anxiety or Stress, All of these)

Symptom frequency: How often are symptoms experienced? (Sometimes/Never/Rarely/Frequently)

Health precautions: Are students taking precautions to mitigate potential health risks? (Taking Break during prolonged use/Using Blue light filter/Limiting Screen Time/None of Above)

Health rating: How would students rate their overall physical and mental health? (Excellent/Good/Fair/Poor)

## Steps  Performed in the Notebook:


``1. Load the Dataset``
The dataset is loaded from a CSV file into a pandas DataFrame for easy manipulation and analysis.

``2. Data Cleaning``
Replace Infinite Values: Any infinite values in the dataset are replaced with NaN to avoid errors in processing.
Handle Missing Values: Missing values in numeric columns are filled with the median value of each column to ensure the dataset is complete and consistent.

``3. Data visualizations``
To understand the data we have used different types of the visualizations like Distribution of the ages, Mobile phone ownership,Plotting frequency of mobile phone use for education,impact of mobile phone use on the academic performance,Distribution of the Health Ratings, performance impact vs Mobile Phone use for the education

``4. Convert Categorical Data``
 Range Conversion: Categorical ranges (e.g., '21-25') are converted to numeric values (e.g., midpoint of the range) to facilitate numerical  operations.
 One-Hot Encoding: Categorical variables are transformed into binary columns (0 or 1) to prepare the data for clustering.
 
``5. Encode Health Ratings Numerically``
The Health rating column is mapped to numeric values (e.g., 4 for 'Excellent', 1 for 'Poor') to integrate this feature into the clustering analysis.

``6. Standardize the Data``
The dataset is standardized to have a mean of 0 and a standard deviation of 1, ensuring all features contribute equally to the clustering process.

``7. Determine Optimal Number of Clusters``
Elbow Method: The optimal number of clusters is determined by plotting the sum of squared distances of samples to their cluster centers against the number of clusters. The point where adding more clusters results in diminishing returns is considered optimal.

``8. Apply KMeans Clustering``
KMeans clustering is applied using the optimal number of clusters, and each student is assigned to a cluster based on their feature values.

``9. Visualization``
Elbow Plot: This plot visualizes the results of the elbow method, showing how inertia (sum of squared distances) changes with the number of clusters.

Pairplot: This plot visualizes the clustering results, displaying how students are grouped based on features like Health rating and Age, with different clusters represented in different colors.

## How to Run the Code?

1.Ensure you have the required dependencies installed.
2.Place the dataset (data.csv) and the jupyternotebook (Kddproject) in the same directory.
3.click the run button on the jupyter notebook environment or in the 

The script will perform data preprocessing, apply clustering, and generate visualizations.

# Interpretation

The generated visualizations will help you understand how different features, such as health ratings and age, contribute to clustering students based on their data. This can provide insights into the relationship between students' health and their academic performance.

# Example visulaizations

![Alt text](https://github.com/saiharsha009/KDD-Project-Grp11/blob/main/pie.png?raw=true)
![Alt text](C:\Users\Dell\Desktop\kdd\healthratings.png)

# Deliverable 2

## Data Preparation

- Replace Infinite Values: Any infinite values in the dataset are replaced with NaN to avoid errors in processing.
Handle Missing Values: Missing values in numeric columns are filled with the median value of each column to ensure the dataset is complete and consistent.

- Apply KMeans Clustering
  KMeans clustering is applied using the optimal number of clusters, and each student is assigned to a cluster based on their feature values.

- Visualization
Elbow Plot: This plot visualizes the results of the elbow method, showing how inertia (sum of squared distances) changes with the number of clusters.

- Pairplot: This plot visualizes the clustering results, displaying how students are grouped based on features like Health rating and Age, with different clusters represented in different colors.

![Alt text](C:\Users\Dell\Desktop\kdd\elbow.png)

![Alt text](C:\Users\Dell\Desktop\kdd\clusters.png)

### Converting and Standardizing the "Health rating"

*Standardizing the 'Health rating' column involves grouping comparable ratings into more comprehensive categories. More specifically, "Good;Fair" becomes "Fair" while "Excellent;Good" becomes "Excellent." The make_numerics function is used to turn these ratings into numerical values after standardisation and generates the numeric mapping. For verification, the generated numerical values and their mappings are printed.

After the standazing we plotted a distribution table for the health rating

![Alt text](C:\Users\Dell\Desktop\kdd\standard.png)

## Splitting Data into Training and Testing Sets

- First, we extract the target variable 'Health rating' from the DataFrame and store it in `y`,
- then remove 'Health rating' from the DataFrame.
- The remaining DataFrame is converted to a numpy array for use as features (`X`).
- We then delete the original DataFrame to clean up. The data is split into training (80%) and testing (20%) sets 
- using `train_test_split`, ensuring a reproducible split with a specified random state.
- Intermediate variables are deleted to free up memory, and the shapes of the resulting datasets are printed to confirm the split.

To prepare for model training and evaluation with PyCaret, we start by combining our training features (X_train) with the target variable (y_train). This process creates a unified DataFrame that includes both our feature columns and the target column labeled 'Health rating'.

## Model Training, Evaluation, and Comparison with PyCaret

- With this consolidated DataFrame, we proceed to initialize PyCaret. We configure the setup to include normalization, which standardizes our features for better model performance. We also set n_jobs to 1, which ensures that only one core is used for parallel processing, optimizing resource management and preventing potential conflicts.

- Once the setup is complete, we create and compare different models. For this exercise, we'll focus on two popular classifiers:

- Logistic Regression (log_reg): This model is a classic choice for classification tasks, known for its simplicity and interpretability. It evaluates the relationship between the target variable and features, providing probabilistic predictions.

- Random Forest Classifier (rf): This model leverages an ensemble of decision trees to improve predictive performance and handle complex data structures. It’s robust to overfitting and excels in handling diverse feature sets.

- After training the models, we use PyCaret's evaluate_model function to assess their performance visually. This step helps us understand how well each model performs across various metrics and scenarios.

- Finally, we print out the performance metrics for both models. This comparison allows us to gauge their effectiveness and determine which model better meets our needs for predicting the 'Health rating'.

## Evaluation: (which method provided the most accuracy/best results

-for the evalution part we got KNeighbourClassifier has more accuracy
-In the best performing model 
*KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                     weights='uniform')
                     
## Conclusion/Results:  (what did you learn)
*1.The results and data analysis reveal some crucial insights into the performance of various categorisation methods. The K Neighbours Classifier (KNN) achieves 50% accuracy and has an AUC of 0.4317. While it does pretty well in terms of classification accuracy, its AUC and other metrics like as precision and recall are less spectacular, indicating that it fails to discriminate between classes efficiently.

*2.The Extra Trees Classifier performs similarly, with 50% accuracy and an AUC of 0.4517. It exhibits higher precision and recall than KNN, demonstrating improved performance in specific areas but still struggles with overall class separation.

*3.Gradient Boosting Classifier and Light Gradient Boosting Machine (LightGBM) produce inconsistent results. Both have accuracy values close to 50%, however their AUC scores are lower, with Gradient Boosting at 0.0000 and LightGBM at 0.4417. This shows that, despite their high accuracy, these models may be ineffective at differentiating across classes.

*4.The Random Forest Classifier has a 44% accuracy rate and an AUC of 0.4975. Despite its balanced accuracy, it has negative Kappa and MCC ratings, showing problems with prediction agreement and quality.

*5.Naive Bayes performs the poorest, with an AUC of 0.3542 and an accuracy of only 22%, illustrating its limits on this dataset.

*6.overall,The analysis of various classification models reveals that none performed exceptionally well across all metrics. The K Neighbors Classifier, while having an accuracy of 50%, struggled with other performance indicators like AUC, precision, and recall. The Extra Trees Classifier and Light Gradient Boosting Machine showed similar limitations, with mixed results in accuracy and AUC. Random Forest achieved a balanced accuracy but had negative Kappa and MCC scores, reflecting issues with prediction agreement. Naive Bayes performed the poorest with an accuracy of only 22%. Overall, KNN emerged as the most effective model, but further tuning and feature improvements are needed to enhance overall performance.

![Alt text](C:\Users\Dell\Desktop\kdd\knnheatmap.png)
![Alt text](C:\Users\Dell\Desktop\kdd\heatmap.png)


## Known Issues (problems with predictors, reporting, bias, etc.) 


- Inconsistent Model Performance: The models' performance varied across measures. For example, while KNN achieved 50% accuracy, its AUC was just 0.4317, showing that it struggled to differentiate between classes. Similarly, the Gradient Boosting Classifier exhibited excellent accuracy but a relatively low AUC, indicating variability in how effectively it distinguished between different health ratings.

- Bias and Overfitting: Some models, such as the Naive Bayes classifier, had significantly poor results, with an accuracy of just 22% and poor recall. This shows that it may be overfitting to certain patterns in the training data while failing to generalise successfully with new data.

- Metric Limitations: The low AUC values observed across multiple models suggest that they may not be successful in distinguishing across classes. For example, Logistic Regression has an AUC of 0.0000, indicating that it couldn't differentiate between classes better than random chance.

- Feature Engineering Gaps: The diversity in model performance shows that the present features may not fully capture the key patterns in the data. Improving feature selection might improve the performance of models such as the Random Forest, which has shown inconsistent results.

- Concerns with evaluation metrics: Models such as the Quadratic Discriminant Analysis have negative Kappa and MCC values, suggesting that their predictions were not significantly better than random guessing. This points to the need for improved model tuning or selection.

**To increase model performance and dependability, these challenges must be addressed by refining features, experimenting with new techniques,





