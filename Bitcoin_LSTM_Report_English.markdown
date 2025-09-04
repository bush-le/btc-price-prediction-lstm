# Report: Application of LSTM in Forecasting Bitcoin Price and Trend Fluctuations

## Chapter 1: Introduction and Problem Statement

### 1.1. Background and Urgency of the Topic

The cryptocurrency market, with Bitcoin (BTC) as the leading representative, has become a highly volatile financial sector attracting significant attention from individual investors, financial institutions, and the research community. A prominent feature of Bitcoin is its strong price volatility, influenced by complex factors such as market supply and demand, legal policies, macroeconomic news, and crowd psychology. The complexity and non-linearity of Bitcoin price data pose many challenges for traditional forecasting methods. Therefore, building advanced forecasting models capable of learning deep hidden patterns and time dependencies in the data is extremely important.

Forecasting the value and trend fluctuations of Bitcoin is not merely an academic problem but also brings significant practical value. For investors, accurate forecasting models can support timely trading decisions, effective risk management, and sustainable investment strategies. Therefore, this study focuses on applying the Long Short-Term Memory (LSTM) model, a type of recurrent neural network capable of processing and remembering information in time series, to address two core tasks: price forecasting (regression problem) and trend forecasting (classification problem) of Bitcoin based on historical data.

### 1.2. Research Objectives

The topic is built with the following specific objectives:

- **Data Collection and Preprocessing**: Use multivariate data on Bitcoin from the Binance exchange, including price features, trading volume, and technical indicators, to ensure the quality of input data for analysis and forecasting processes.
- **Data Analysis and Visualization**: Explore the relationships between features and target variables through correlation analysis and Mutual Information techniques, thereby selecting the optimal feature set for each problem.
- **Building Deep Learning Models**: Develop and train two separate LSTM models to solve the price forecasting problem and the increasing/decreasing trend forecasting problem.
- **Evaluation and Optimization**: Use specialized evaluation metrics and visual charts to analyze the effectiveness of each model, thereby drawing conclusions, advantages, limitations, and proposing directions for future improvements.

### 1.3. Overview of the Long Short-Term Memory (LSTM) Model

The Long Short-Term Memory (LSTM) model is a variant of recurrent neural networks (RNN), specifically designed to handle time series data and overcome the shortcomings of traditional RNNs such as vanishing or exploding gradients. The LSTM architecture was introduced by Hochreiter and Schmidhuber in 1997, providing the ability to remember long-term dependencies in data sequences.

Each LSTM unit includes a cell state and three main "gates" that act as information filters:

- **Forget Gate**: Determines which information from the old cell state needs to be discarded.
- **Input Gate**: Determines which new information needs to be added to the cell state.
- **Output Gate**: Determines which part of the information from the cell state will be used to generate the output at the current time step.

These mechanisms allow LSTM to maintain important information over many time steps without losing it, while eliminating unnecessary noise. Thanks to this capability, LSTM becomes a suitable choice for modeling complex and non-linear financial time series like Bitcoin prices, helping the model learn rules and price patterns from 15-minute candle data effectively.

### 1.4. Report Structure

This report is organized into five chapters. Chapter 1 presents the background, reasons for choosing the topic, and an overview of the method. Chapter 2 describes in detail the data processing process, feature analysis, and selection of suitable feature sets for each problem. Chapter 3 presents the architecture and training parameters of the two LSTM models. Chapter 4 delves into the analysis and evaluation of experimental results. Finally, Chapter 5 provides an overall conclusion, highlighting advantages, limitations, and proposing directions for future development.

## Chapter 2: Data Processing and Analysis

### 2.1. Data Preprocessing Process

The data preprocessing process is a foundational step, ensuring the quality of input data for machine learning models. Bitcoin price data was collected from the Binance exchange with a 15-minute timeframe, including 280,253 rows of data from 2017 to August 2025.

The preprocessing process was carried out according to the following main steps:

- **Adding Column Headers**: The raw data from the Binance API initially had no headers, so standard headers (open_time, open, high, low, close, volume,...) were added for easy processing.
- **Removing Unnecessary Columns**: To reduce noise and optimize data, columns that do not have a direct impact on short-term price forecasting, such as quote_asset_volume or number_of_trades, were removed. Important features retained include open_time, price features (open, high, low, close), and volume.
- **Data Checking and Processing**: The data was checked to ensure no missing, invalid, or duplicate values, creating a clean and reliable dataset.
- **Data Type Conversion**: The open_time column was converted from integer type (timestamp) to datetime format and set as the dataset index, facilitating time series analysis.
- **Adding Technical Indicators**: This is an important step to enrich information for the model. Two technical indicators added are the 20-period Moving Average (MA20) and the 14-period Relative Strength Index (RSI14).

  MA20 smooths price data, helping the model identify short-term trends and reduce noise, while RSI14 provides signals about market sentiment, measuring overbought (>70) or oversold (<30) states of the asset, very useful for trend forecasting problems.

The preprocessing process also includes checking outliers using the IQR method. The results show that Bitcoin price values above 100,000 USD are not noise or data errors, but reflections of strong growth periods in 2024-2025. Therefore, these values were not removed because they contain important information about market cycles, which is a key factor helping the model learn the nature of Bitcoin fluctuations.

### 2.2. Correlation Analysis Between Features

Correlation analysis is used to measure the degree of linear relationship between features. The correlation matrix is presented in the form of a heatmap, helping to visualize this relationship.

The feature correlation matrix chart (see Figure 8) shows some notable points:

- **Very High Correlation**: Price features (Open, High, Low, Close) and the MA20 indicator all have very strong linear correlations with each other, with correlation coefficients close to 1. This proves that these variables contain very similar information, all reflecting the same price fluctuation.
- **Low Correlation**: Volume and RSI14 features have very low correlations with price features (coefficients ~0.1 and ~0.03). This shows that Volume and RSI14 are independent features, providing new information not available in price data, enriching the input dataset.

Although high correlation between price features can cause multicollinearity issues for traditional linear regression models, this does not negatively affect the LSTM model. Instead, it helps the deep learning model easily capture overall price trends and intra-candle fluctuations effectively.

### 2.3. Mutual Information (MI) Analysis

Mutual Information (MI) is a more powerful method than linear correlation, as it can measure non-linear relationships between variables and reflect the amount of information a feature provides to the target variable. The higher the MI, the more important the feature is to the problem. MI analysis was performed for both regression and classification problems.

#### 2.3.1. MI Analysis for Regression Problem (Price Forecasting)

For the price forecasting problem, the target variable is Close at time $t+1$. MI analysis (see Figure 7) shows a clear hierarchy in the importance of features:

- Price features (Close, High, Low, Open) and MA20 have very high MI scores, ranging from 2.89 to 3.75. This confirms that historical prices and moving averages are the most important factors providing information to forecast the absolute value of Bitcoin in the future.
- Volume has a significantly lower MI score (0.13), but still contributes a certain amount of information about market strength.
- RSI14 has a very low MI score (0.05), showing it has almost no informational value for the absolute price forecasting problem.

#### 2.3.2. MI Analysis for Classification Problem (Trend Forecasting)

For the trend forecasting problem, the target variable is the increasing/decreasing trend. MI analysis (see Figure 6) provides a significantly different result:

- RSI14 and MA20 are the two features with the highest MI scores, with RSI14 reaching 0.097 and MA20 reaching 0.034. This proves that these technical indicators are the most important features for predicting Bitcoin fluctuation trends.
- Price features (Open, High, Low, Close) have very low MI scores, ranging from 0.0073 to 0.011.

The clear difference between MI results for the two problems shows that the nature of the problem determines feature selection. Value forecasting (regression) requires information about past prices themselves, while trend forecasting (classification) depends more on indicators of momentum and market strength, reflected through RSI and MA20.

### 2.4. Summary and Feature Selection

Based on the results of correlation and Mutual Information analysis, the optimal feature set was selected for each problem to maximize model performance.

**Table 2.1: Feature Selection for Models**

| Problem                      | Selected Features              | Reason                                                               |
|------------------------------|--------------------------------|----------------------------------------------------------------------|
| Price Forecasting (Regression) | Open, High, Low, Close, MA20, Volume | High correlation and MI with the price target variable, providing sufficient historical price information to forecast absolute values. |
| Trend Forecasting (Classification) | RSI14, MA20, Close            | RSI14 and MA20 have the highest MI, providing strong signals about momentum and market trends, key factors for the classification problem. |

After feature selection, the data was split into training and testing sets with an 80/20 ratio. Finally, the data was normalized using the MinMaxScaler method to bring it to a range. This step is necessary because features have different scales (BTC prices in tens of thousands of USD, RSI from 0-100, large fluctuating Volume), normalization helps the model learn more effectively and avoids gradient vanishing/exploding phenomena.

## Chapter 3: Building and Training LSTM Models

### 3.1. Time Series Data Creation Technique (Sliding Window)

The LSTM model is designed to handle time series data. To provide suitable input, the sliding window technique was applied. This technique divides the historical data sequence into subsequences, each subsequence including a number of time steps (timesteps) and corresponding features. The data is converted from 2D form ([samples, features]) to 3D form ([samples, timesteps, features]), which is the standard input format for the LSTM model.

The sliding window size was chosen differently for each problem:

- **Regression Problem**: 32-step sliding window (equivalent to 8 hours of data).
- **Classification Problem**: 96-step sliding window (equivalent to 24 hours of data).

This difference stems from the nature of each problem. Absolute value forecasting (regression) may depend heavily on the most recent developments, while trend forecasting (classification) requires a broader context to identify oscillation patterns and market sentiment.

### 3.2. Configuration and Training Parameters for Regression Problem

The LSTM model built for the price forecasting problem has a multi-layer architecture, designed to extract features and predict continuous values:

- **Layer 1**: LSTM(128) with return_sequences=True, to return the entire hidden sequence so the next layer can learn more detailed features.
- **Dropout Layer**: Dropout(0.3), to randomly drop 30% of neurons during training to minimize overfitting.
- **Layer 2**: LSTM(64) with return_sequences=False, focusing on learning deeper features and returning only the final state of the sequence.
- **Fully-connected Layer**: Dense(32, activation='relu'), to extract non-linear features.
- **Output Layer**: Dense(1), with a single neuron to predict the continuous value of the closing price.

The model was compiled with the Mean Squared Error (MSE) loss function and Adam optimization algorithm, with a learning rate of 0.0001. The training process uses EarlyStopping callbacks to stop early when validation loss does not improve after 10 epochs, and ReduceLROnPlateau to automatically reduce the learning rate when the model plateaus after 5 epochs. These callbacks help optimize the training process and ensure the model achieves the highest performance. The model was trained for 200 epochs with a batch size of 64.

### 3.3. Configuration and Training Parameters for Classification Problem

Similar to the regression model, the classification model also has a similar architecture but adjusted for the trend prediction problem:

- **LSTM Layer 1**: LSTM(128) with return_sequences=True.
- **Dropout Layer**: Dropout(0.3).
- **LSTM Layer 2**: LSTM(64) with return_sequences=False.
- **Fully-connected Layer**: Dense(32, activation='relu').
- **Output Layer**: Dense(1) with sigmoid activation function, allowing the model to predict probabilities for two classes (increase or decrease).

The classification model uses the binary_crossentropy loss function and Adam optimization algorithm with a learning rate of 0.0001. EarlyStopping and ReduceLROnPlateau callbacks were also used to optimize the training process. The model was trained for 150 epochs with a batch size of 64.

## Chapter 4: Results and Model Evaluation

### 4.1. Price Forecasting Results (Regression Problem)

#### 4.1.1. Analysis of Loss Chart Over Epochs

The loss chart over epochs (see Figure 5) shows that the regression model converged very well and was not overfitting. Both training loss and validation loss decreased rapidly in the early epochs and continued to decrease gradually, in parallel, until reaching a very low level (near 10−4). This convergence proves that the model has learned the basic rules from the training data and has good generalization ability on new data. The training process stopped early at epoch 98 thanks to the EarlyStopping mechanism, showing that the model has achieved optimal performance without needing to run all the defined epochs.

#### 4.1.2. Evaluation of Prediction Chart Compared to Actual Price

The chart comparing predicted price and actual price (see Figure 4) clearly illustrates the superior performance of the model. The predicted price line (orange) almost perfectly matches the actual price line (blue), closely following all fluctuations and main trends in the testing period.

However, upon deeper analysis, a small limitation can be seen: at price peaks (local maxima) and troughs (local minima), the model tends to "smooth" and predict lower than the peak or higher than the trough. This phenomenon is common in time series models, as they react slower to sudden and extreme fluctuations. Nevertheless, the model's ability to capture the overall trend is still very strong, making it a reliable support tool.

#### 4.1.3. Error Distribution Analysis

The error distribution chart (True - Predicted) (see Figure 3) shows that most of the model's errors are tightly concentrated around the value 0. This distribution has a shape similar to a Gaussian normal distribution, with common errors in the range of ±300−400 USD. This shows that the model operates stably and predictions have small, random errors.

The appearance of a few larger errors (>1000 USD) at the two ends of the chart may correspond to times when the market has extremely strong fluctuations, when the model has difficulty predicting accurately.

Overall, the regression model's evaluation metrics also confirm high accuracy:

- **MSE (Mean Squared Error)**: 167,996.99
- **RMSE (Root Mean Squared Error)**: 409.87 USD
- **MAE (Mean Absolute Error)**: 294.18 USD
- **R2 Score (Coefficient of Determination)**: 0.98

With an R2 score of 0.98, the model can explain up to 98% of Bitcoin price fluctuations in the test set. The mean absolute error (MAE) is only about 294 USD, a very small number compared to the average price above 100,000 USD, equivalent to an error of only about 0.3-0.4%.

### 4.2. Trend Forecasting Results (Classification Problem)

#### 4.2.1. Analysis of Loss Chart Over Epochs

The training loss and validation loss chart for the classification problem (see Figure 2) shows a much poorer training performance compared to the regression model. Although training loss decreases steadily, validation loss does not decrease deeply and has strong fluctuations, showing that the model is having difficulty learning and generalizing on unseen data. This signals that the model may be underfitting, meaning it is not strong enough to capture the complex features of the trend prediction problem.

#### 4.2.2. Confusion Matrix Evaluation

The Confusion Matrix (see Figure 1) provides a detailed view of the classification model's performance:

- **TP (True Positive)**: 770 (Predicted increase, actual increase)
- **FP (False Positive)**: 331 (Predicted increase, actual decrease)
- **TN (True Negative)**: 1749 (Predicted decrease, actual decrease)
- **FN (False Negative)**: 1496 (Predicted decrease, actual increase)

Analysis of the matrix shows that the model has a bias towards predicting "decrease" (class 0). Although capable of correctly predicting decrease cases (TN = 1749), the model misses a large number of actual increase cases (FN = 1496).

The evaluation metrics also reflect this poor performance:

- **Accuracy**: 0.58 (Only slightly higher than random prediction 0.5)
- **Precision (class 1 - increase)**: 0.70 (When the model predicts increase, accuracy is 70%)
- **Recall (class 1 - increase)**: 0.34 (The model only detects 34% of actual increase sessions)
- **F1-score (class 1 - increase)**: 0.46 (Low, showing imbalance between precision and recall)

In particular, the low recall shows that the model missed nearly 2/3 of actual price increase opportunities, which significantly reduces the model's usefulness in a trading environment.

## Chapter 5: Conclusion and Development Directions

### 5.1. Summary of Achieved Results

The study has demonstrated the feasibility of applying the LSTM model in forecasting financial time series, specifically Bitcoin prices and trend fluctuations.

- **Regression Model (Price Forecasting)** achieved excellent results, with small, stable errors and the ability to explain price fluctuations up to 98% (R2=0.98). This model has high application potential in supporting short-term trading decisions.
- **Classification Model (Trend Forecasting)** gave less promising results, with accuracy only 58% and low recall (0.34), showing the model still has many limitations in detecting actual price increase opportunities.

### 5.2. Evaluation of Advantages and Limitations

**Advantages**:

- **Suitability of LSTM**: The report affirms that the LSTM model is very effective in handling complex data sequences with long-term dependencies like Bitcoin prices.
- **Practical Value**: The regression model has high accuracy, can be used as a reliable support tool in technical analysis and risk management.
- **Multivariate Data Processing Capability**: Combining price features, volume, and technical indicators has helped the model exploit many aspects of the market.

**Limitations**:

- **Peak/Trough Prediction Capability**: The regression model still has limitations in reacting immediately to extreme fluctuations, often predicting lower than peaks and higher than troughs.
- **Classification Performance**: The trend classification model is not reliable enough to provide independent trading signals, due to low performance and bias towards one class.
- **Data Limitations**: Current models only rely on historical price data, not integrating external factors such as news, market sentiment, and macroeconomic events, which have a significant impact on Bitcoin prices.

### 5.3. Directions for Improvement and Future Development

Based on the identified limitations, future development directions can focus on the following points:

- **Expanding Feature Set**: Integrate additional technical indicators such as MACD, Bollinger Bands, Stochastic, along with unstructured data like sentiment analysis from news, social media posts, so the model can "understand" market context and forecast sudden fluctuations more accurately.
- **Optimizing Classification Model**: Need to address the imbalance in predicting classes by using data balancing techniques (oversampling/undersampling) or more suitable loss functions like focal loss.
- **Testing Other Model Architectures**: Compare LSTM performance with other deep learning architectures like GRU (Gated Recurrent Unit) or Transformer, which are very successful in data sequence processing problems, to find a more optimal model for financial time series forecasting problems.

### 5.4. Overall Summary

In summary, the study has achieved important accomplishments in applying LSTM to forecast Bitcoin prices, opening up a promising approach for analyzing and forecasting the cryptocurrency market. Although the classification model still needs many improvements, the excellent results of the regression model have provided a powerful tool that can be applied immediately in practice, while serving as a solid foundation for more comprehensive future studies.