# DataMining2022
Hands-on experience for: 1. Feature Extraction 2. EDA 3. Time Series Forecasting 4. Word-Embedding 5. DL sentiment analysis

## HW1. Decision Tree

冰與火之歌資料集：
https://www.kaggle.com/mylesoneill/game-of-thrones

預測角色是否死亡 

Death Chapter 作為預測目標

1. 欄位的空值轉成0(代表存活)，有數值的轉成1(代表死亡)

2. 將Allegiances轉成dummy特徵(底下有幾種分類就會變成幾個特徵，值是0或1，本來的資料集就會再增加約20種特徵)

3. 亂數拆成訓練集(75%)與測試集(25%) 

4. 使用scikit-learn的DecisionTreeClassifier進行預測

5. 做出Confusion Matrix，並計算Precision, Recall, Accuracy 

6. 產出決策樹的圖

## HW2. sentiment analysis

1. 資料前處理

a. 讀取 csv 檔後取前 1 萬筆資料

僅保留"Text"、"Score"兩個欄位

並將 "Score" 欄位內值大於等於4的轉成1，其餘轉成0

1: positive

0: negative

並將text欄位內的文字利用分割符號切割

b. 去除停頓詞stop words 

c. 文字探勘前處理，將文字轉換成向量，請實作 tf-idf 及 word2vec 並進行比較


2. 建模：

a. 使用 Random forest進行分類

b. 請寫自行撰寫function進行k-fold cross-validation(不可使用套件)並計算Accuracy

            b-1. input(k, data)，將data切成k份，其中1份當測試集，剩餘k-1份當訓練集建立模型

            b-2. 輪流將k份的每份資料都當 測試集，其餘當訓練集建立模型，因此會進行k次，k次都計算出Accuracy

            b-3. 將k次的Accuracy平均即為output

3. 評估模型


## HW3. Time Series Regression(新竹PM2.5預測)

將前六小時的汙染物數據做為特徵，未來第一個小時/未來第六個小時的pm2.5數據為預測目標
使用兩種模型 Linear Regression 和 XGBoost 建模並計算MAE

1. 資料前處理

 a. 取出10.11.12月資料

 b. 缺失值以及無效值以前後一小時平均值取代 (如果前一小時仍有空值，再取更前一小時)

 c. NR表示無降雨，以0取代

 d. 將資料切割成訓練集(10.11月)以及測試集(12月)

 e. 製作時序資料: 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料

     **hint: 將訓練集每18行合併，轉換成維度為(18,61*24)的DataFrame(每個屬性都有61天*24小時共1464筆資料)

2. 時間序列

  a.預測目標

     1. 將未來第一個小時當預測目標

     2. 將未來第六個小時當預測目標

 b. X請分別取

     1. 只有PM2.5 (e.g. X[0]會有6個特徵，即第0~5小時的PM2.5數值)

     2. 所有18種屬性 (e.g. X[0]會有18*6個特徵，即第0~5小時的所有18種屬性數值)

 c. 使用兩種模型 Linear Regression 和 XGBoost 建模

 d. 用測試集資料計算MAE (會有8個結果， 2種X資料 * 2種Y資料 * 2種模型)

## HW4. sentiment analysis 2

1. 資料處理與HW2同

2. 建模

a. 分別用CNN與LSTM對train的資料進行建模，可自行設計神經網路的架構

b. 加入Dropout Layer設定Dropout參數(建議0.7)進行比較

c. plot出訓練過程中的Accuracy與Loss值變化

3. 評估模型

a. 利用kaggle上test的資料對2.所建立的模型進行測試，並計算Accuracy
