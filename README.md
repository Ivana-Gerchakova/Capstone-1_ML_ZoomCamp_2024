# Capstone-1_ML_ZoomCamp_2024

## Project Overview :

### Space Mission Prediction
Predicting the success rate of space missions using machine-learning models. 
This project explores various models to analyze and predict factors influencing the success of space missions. 
By leveraging features such as mission cost, duration, and distance from Earth,the models aim to provide actionable insights for optimizing mission planning and resource allocation. 

**Source**: The data is sourced from the Space Missions Dataset on Kaggle. [Kaggle - Space Missions Dataset](https://www.kaggle.com/datasets/sameerk2004/space-missions-dataset)

<div align="center">
    <img src="images/images 1.jpg" />
</div>

## Problem Description: 

Space missions involve significant complexity and high costs, making their success crucial for scientific advancement and resource efficiency. 
Accurately predicting the success rate of missions is essential for informed decision-making and risk management. 
This project leverages historical space mission data to identify critical factors driving mission success, enabling a data-driven approach to optimize planning and execution.

The goal is to develop a machine-learning model that can accurately predict mission success percentages based on 
features like mission cost, duration, distance from Earth, and scientific yield.
This solution has the potential to optimize decision-making processes in future space missions and reduce the risk of failures.

## Dataset Details

The dataset used in this project is the **Space Missions Dataset**, sourced from Kaggle. It contains over 500 records, with detailed information on historical space missions.

### **Key Features:**
- **Mission Type:** The type of the mission (e.g., Scientific, Commercial, Military, etc.).
- **Mission Cost (billion USD):** The total cost of the mission.
- **Mission Success (%):** The success rate of the mission, given as a percentage.
- **Distance from Earth (light-years):** Distance to the mission's target.
- **Scientific Yield (points):** The scientific outcome of the mission.
- **Mission Duration (years):** Duration of the mission in years.
- **Crew Size:** The number of astronauts involved in the mission.
  
<div align="center">
    <img src="images/images 2.jpg" />
</div>

## EDA (Exploratory Data Analysis)

Exploratory Data Analysis (EDA) was conducted to gain a deeper understanding of the dataset, uncover relationships between features, identify anomalies, and prepare the data for modeling. 
Below are the key steps and insights derived during the analysis:

### Key Steps:
1. **Handling Missing Values and Dropping Unnecessary Columns:**
   - Checked for missing values across all features and handled them.
   - Dropped unnecessary columns that were irrelevant to the analysis.

2. **Outlier Detection and Removal:**
   - Detected and removed outliers using the Interquartile Range (IQR) method to improve model robustness and accuracy.

3. **Descriptive Statistics:**
   - Calculated summary statistics (mean, median, min, max, standard deviation) for numerical features to understand their distributions.

4. **Correlation Analysis:**
   - Generated a correlation heatmap to identify relationships between numerical features and their potential impact on mission success.

5. **Visualization of Distributions:**
   - Utilized various visualizations, including:
     - **Histograms** and **boxplots** for numerical feature distributions.
     - **Pie charts** to analyze target type distribution.
     - **Bar plots** and **countplots** for categorical features.
     - **Scatter plots** for relationships between numerical variables.
     - **Bar charts** for visualizing top mission names.

6. **Feature Importance Analysis:**
   - Analyzed feature importance using multiple techniques:
     - **Regression model coefficients**
     - **Recursive Feature Elimination (RFE)**
     - **Permutation Importance**
     - **Drop-Column Importance**

### Visual Insights:
1. **Mission Success Distribution:**  
   - Visualized the distribution of target types using a pie chart.
     
   ![Mission Success Distribution](images/png2.png)

2. **Feature Importance:**  
   - Displayed feature importance scores derived from various analysis methods.
     
   ![Feature Importance](images/png1.png)

### Outcome:
EDA provided critical insights into the dataset, including feature relationships and distributions, enabling effective feature engineering and selection. 
These insights laid the foundation for robust and accurate predictive modeling.

