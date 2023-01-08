# Predicting Coupon Redemption

------

*For steps that do not include detailed code implementation, please refer to the Appendix.*

## 1 Introduction

### 1.1 Background Information

ABC is an established Brick & Mortar retailer. It frequently conducts discount marketing campaigns, which offer coupons for a specific product/range of products, to attract new customers and to retain and reinforce loyalty of existing customers. The entire coupon redemption process could be summarized below:

1. Customers receive coupons issued by ABC under various campaigns from various channels including email, notifications, etc.
2. Customers can choose whether to redeem the received coupon within the duration between campaign start date and end date.
3. If the coupon is redeemed, the customer can purchase products covered at a discounted price at the ABC's retailer store.

*The redemption process is shown below:*

![image](https://user-images.githubusercontent.com/62502750/211182156-cf88e1f3-2d95-4e5a-ad99-d5b379900a78.png)

### 1.2 Problem Statement

Since the redemption rate of previous campaigns are not satisfactory, ABC company is expecting a model to predict customer redemption behavior based upon the datasets they provided, which can help tremendously to construct more effective and powerful discount marketing strategies. In addition to this model, we also offer to attract more business insight from the dataset and aim to provide guideline for campaign design. 

In summary, this project is aimed for the following two goals:

1. Provide a general guideline for future discount marketing strategies from the perspective of campaigns, items and customers.

2. Construct a model to accurately predict whether a customer will redeem a specific coupon of a certain campaign.


### 1.3 Data Understanding

#### 1.3.1 Dataset description

The datasets available contains information and details about campaign-coupon-customer mapping, customer demographic, customer transaction, coupon-item mapping and items.

1. **train:** Train data containing the coupons offered to the given customers under the 18 campaigns. Note that the label *y* is the attribute *redemption_status*, where 0 indicates *coupon not redeemed* and 1 indicates *coupon redeemed*.

2. **campaign:** Campaign information for each of the 18 campaigns. Notes ABC retailer designs two different types of campaign, denoted as X/Y.

3. **coupon item mapping:** Mapping of coupon and items valid for discount under that coupon.

4. **customer demographics**: Customer demographic information for some customers.

5. **customer transaction:** Transaction data for all customers for duration of campaigns in the train data. Note that the used readapted coupon reflects in the attribute *coupon_discount*, and the *selling_price* is the total sales value of one transaction of a specific item. 

6. **item:** Item information for each item sold by the retailer.

7. **test:** Contains the coupon customer combination for which redemption status is to be predicted. (Without label)

![image](https://user-images.githubusercontent.com/62502750/211182184-337bfc64-14c8-48ae-bb9a-479385bc4376.png)

#### 1.3.2 Scope of Work

Based on the dataset given, we aim to:

1.  predict the probability of coupon redemption for each campaign, coupon and customer combination. 

2.  attract business insights of discount marketing strategies  from the perspective of campaigns, items and customers.


To achieve the above-mentioned objectives we propose the following scope of work:

      1. Predict coupon redemption status by a classification model .
      2. Predict campaign performance by a regression model.
      3. Discover strongly correlated itemset with corresponding discount information by frequent pattern mining.
      4. Conduct customer segmentation and discover the pattern in customer redemption behavior by clustering model.

## 2 Methodology

### 2.1 Regression

#### 2.1.1 Data Pre-Processing

By summarizing the *train* dataset, we calculate the redemption ratio for each campaign as an indicator of the campaign performance.

| campaign_id | redemption_ratio% |
| :---------: | :---------------: |
|      1      |     0.699301      |
|      2      |     0.651042      |
|      3      |     0.490196      |
|      4      |     0.720165      |
|      5      |     0.383142      |
|      6      |      1.53846      |
|      7      |     0.252525      |
|      8      |     0.882288      |
|      9      |      0.72601      |
|     10      |     0.406268      |
|     11      |     0.215672      |
|     12      |     0.470588      |
|     13      |      1.53057      |
|     26      |     0.858653      |
|     27      |     0.308642      |
|     28      |     0.210084      |
|     29      |     0.410783      |
|     30      |     0.603062      |

In order to predict the campaign performance, which is represented by the redemption ratio, we summarize information from dataset *campaign* and *train* and discover several potential predictors, including campaign type, duration, the total amount of coupons, customers covering ratio, coupon covering ratio, and the number of coupons distributed every day. 

- Campaign type: all the campaign is divided into two types, X and Y. This variable value equals 1 if the campaign type is X, and the value equals 0 if the campaign type is Y.

- Duration (days): the length of time from the start to the end of the campaign.

- Total Amount(of coupons): the total number of coupons issued in one campaign.

- Customers covering ratio: how many different customers are covered in one campaign divided by the total number of different customers.

- Coupon covering ratio: how many unique coupons are covered in one campaign divided by the total number of unique coupons.

- Amount (of coupons distributed) /Day: the total amount of coupons divided by duration.

```python
import numpy as np
import pandas as pd

train = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/train.csv")
transaction = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/customer_transaction_data.csv", parse_dates=['date'])

d1 = train.groupby('campaign_id').agg(redemption_frequency=('redemption_status','sum'))
d2 = train['campaign_id'].value_counts().sort_index().rename('total_amount').to_frame()
d2.index.name = d1.index.name
d3 = train.groupby('campaign_id')['coupon_id'].nunique()
coupon_types = train.coupon_id.nunique()
d3 = (d3 / coupon_types * 100).rename('coupon_covering_ratio%')
d4 = train.groupby('campaign_id')['customer_id'].nunique()
n_customers = train.customer_id.nunique()
d4 = (d4 / n_customers * 100).rename('customer_covering_ratio%')

d = pd.concat([d1, d2, d3, d4], axis=1)
d['redemption_ratio%'] = d['redemption_frequency'] / d['total_amount'] * 100
d = d.drop(columns=['redemption_frequency'])

for index, row in campaign.iterrows():
  if row.end_date < row.start_date:
    campaign.iloc[index, 2] = row.end_date
    campaign.iloc[index, 3] = row.start_date

    campaign['duration_days'] = (campaign['end_date'] - campaign['start_date']).dt.days

campaign['start_year'] = campaign['start_date'].dt.year
campaign['start_quarter'] = campaign.start_date.dt.quarter
campaign['start_week'] = campaign.start_date.dt.isocalendar().week
campaign['start_weekday'] = campaign.start_date.dt.weekday

campaign['end_year'] = campaign['end_date'].dt.year
campaign['end_quarter'] =  campaign.end_date.dt.quarter
campaign['end_week'] =  campaign.end_date.dt.isocalendar().week
campaign['end_weekday'] =  campaign.end_date.dt.weekday
df = campaign.drop(columns=['start_date', 'end_date'])
df = df.merge(d, on='campaign_id', how='right')

df['amount/day'] = df['total_amount'] / df['duration_days']
df["redemption_ratio"] = df['redemption_ratio%']
df = df.drop(columns = ['redemption_ratio%'])
df.rename(columns={"redemption_ratio":'redemption_ratio%'}, inplace=True)
```

We also detect that there is an outlier in the data, since in our dataset it is seldom to see a campaign continues for 289 days. In addition, data transformation and normalization are implemented and features with high correlation are removed.

#### 2.1.2 Model Development and Evaluation

After conducting stepwise regression analysis, we find that *campaign type* and *duration(days)* are significant predictors for label *redemption ratio*. The regression result can be seen in the following graph.

![image](https://user-images.githubusercontent.com/62502750/211182198-3309ebd0-6d02-4ff1-ba77-29932b2b4a90.png)

The regression model is: $ E(y)=0.4656+0.4919x_1-0.3397x_2 $. The $p$-value of both the individual $t$-test and global $F$-test is significant at 5% significance level, which means this model is useful for estimating campaign performance and all predictors are significant.

#### 2.1.3 Analysis and Business Insight

From this model we can conclude that:

1. The average redemption rate of X type campaigns is significantly larger than that of Y.

2. On average, shorter campaign duration have higher redemption rate, within the experiment region 12 days – 182 days.

3. The model is useful to estimate campaign performance for future discount marketing.

   

### 2.2 Frequent Pattern Mining

Association rules reveal the correlation between itemset and can be useful for constructing recommendation or marketing strategies. As an example, offering package discounts for products with stronger association rules can potentially give customers more incentive to redeem coupons, thereby increasing the redemption rate.

Thus, we conduct frequent pattern mining to discover strong association rules and correlated itemset.

#### 2.2.1 Data Pre-Processing

Firstly, we group unique items that are purchased by same customer on the same date (dataset *customer transaction*) as one transaction record. 

```python
import numpy as np
import pandas as pd

transaction = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/customer_transaction_data.csv", parse_dates=['date'])
fp_item = transaction.groupby(['date', 'customer_id'])['item_id'].unique().reset_index(drop=True)
```



#### 2.2.2 Model Selection and Development

**FP-Growth Algorithm Implementation**

Since we have over 140,000 transaction record, it will be time- and computation-consuming to scan the whole data multiple times. Thus we implement FP-Growth instead of Apriori, since FP-Growth algorithm mines in the conditional pattern base to avoid generating candidate itemsets and reduce the search size substantially.

Association rules and correlated itemsets are also derived from the result of FP-Growth. Here, we define correlated itemsets as a pair of frequent itemset $A, B$ with $\text{confidence}(A ⇒ B)$ and $\text{confidence}(B ⇒ A)$ both above the min confidence threshold.

```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from scipy import stats

def frequent_pattern(df, min_support:float):
  '''
  Input a transactions series and minimal support.
  Return a frequent pattern DataFrame default by fptree algorithm. 

  '''
  te = TransactionEncoder()
  te_array = te.fit(df).transform(df)
  d = pd.DataFrame(te_array, columns=te.columns_)
  fp = fpgrowth(d, min_support=min_support, use_colnames=True)
  
  return fp

def association(df, min_support:float, min_confidence:float):
  '''
  Input a transactions series, minimal support and minimal confidence.
  Return a association rules DataFrame.
  '''
  n=len(df)
  ar = association_rules(frequent_pattern(df, min_support), metric='confidence', min_threshold=min_confidence).drop(columns=["lift","leverage"])
  chi_square(ar, n)
  confidences(ar)
  conviction(ar)
  imbalance_ratio(ar)
  ar = ar.loc[:,["antecedents", "consequents", "antecedent support", "consequent support", "support", "confidence", "confidence_inverse", \
                                         "conviction","conviction_inverse", "lift", "chi_square", "p_value", "leverage", "all_confidence", \
                                                                               "max_confidence", "kulczynski", "cosine", "imbalance_ratio"]]
  return ar

def correlation(df, min_support:float, min_confidence:float):
  '''
  Input a transactions series, minimal support and minimal confidence.
  Return a correlation DataFrame.
  '''
  n = len(df)
  corr = association(df, min_support, min_confidence)
  corr = corr.loc[corr.all_confidence>=min_confidence, :]
  drop_duplicate_association(corr)

  return corr
```

**Grid Search for Thresholds**

In order to determine the min support so that there are enough frequent $k$-itemsets ($k>1$) to mine the association rules, we conduct grid search to experiment on different values of min support and derive the following table.

```python
n_fp = []
n_1itemset = []
for i in range(1, 11):
  min_supp = i/1000
  fp = frequent_pattern(fp_item, min_supp)
  fp_1item = fp[fp.itemsets.apply(len)==1]
  n_fp.append(len(fp))
  n_1itemset.append(len(fp_1item))

min_support = np.array(range(1, 11))/1000
df = pd.DataFrame({'min_support':min_support, "number of frequent patterns":n_fp, "number of frequent 1-itemset":n_1itemset})
df["number of frequent k-itemset(k>1)"] = df.iloc[:,1]-df.iloc[:,2]
```

| **min_support** | **no. of** **fq** | **no. of frequent 1-itemset** | **no. of frequent k-itemset(k>1)** |
| :-------------: | :---------------: | :---------------------------: | :--------------------------------: |
|      0.001      |       1664        |             1598              |                 66                 |
|      0.002      |        474        |              467              |                 7                  |
|      0.003      |        217        |              216              |                 1                  |
|      0.004      |        127        |              127              |                 0                  |
|      0.005      |        81         |              81               |                 0                  |
|      0.006      |        50         |              50               |                 0                  |

The table indicates that min support of 0.001 and 0.002 have relatively large number of frequent $k$-itemsets ($k>1$). This is relatively a small value due to large amounts of transactions (142,445) available. In real-life data, the relative frequency of frequent patterns are highly affected by the number of null transactions in the dataset.

Apart from min support, associated rules will be deemed uninteresting if they do not satisfy the min confidence threshold. Thus, we narrowed min support down to 0.001 and 0.002 and conduct grid search for the value of min confidence.

```python
obs = []
for i in (0.001, 0.002):
  for j in range(20, 65, 5):
    min_confidence = j/100
    n_ar = len(association(fp_item, i, min_confidence))
    n_corr = len(correlation(fp_item, i, min_confidence))
    obs.append({"min_support":i, "min_confidence":min_confidence, "number of association rules":n_ar, "number of correlations":n_corr})
    
df2 = pd.DataFrame(obs)
```

| **min_support** | **min_confidence** | **no. of association rules** | **no. of correlations** |
| :-------------: | :----------------: | :--------------------------: | :---------------------: |
|      0.001      |        0.2         |              52              |           21            |
|      0.001      |        0.25        |              43              |           16            |
|      0.001      |        0.3         |              32              |            8            |
|      0.001      |        0.35        |              20              |            5            |
|      0.001      |        0.4         |              10              |            1            |
|      0.001      |        0.45        |              3               |            1            |
|      0.001      |        0.5         |              0               |            0            |
|      0.001      |        0.55        |              0               |            0            |
|      0.001      |        0.6         |              0               |            0            |
|      0.002      |        0.2         |              6               |            2            |
|      0.002      |        0.25        |              4               |            1            |
|      0.002      |        0.3         |              2               |            0            |
|      0.002      |        0.35        |              1               |            0            |
|      0.002      |        0.4         |              1               |            0            |

We set min confidence ranging from 0.2 to 0.6 with min support of 0.001 or 0.002, and we find that when min support equals 0.002 there will be no correlated itemset when min confidence is above 0.25. Since confidence indicates the probability of having the consequent ($B$) given the antecedent ($A$), the min confidence threshold cannot be too small otherwise the association rule would be too weak. Thus, the min support is determined as 0.001. Since the number of association rules and correlated itemset decrease to 0 when min confidence increases up to 0.5, we recommend the min confidence to be no less than 0.3 and no more than 0.45.

#### 2.2.3 Model Evaluation 

A challenge in mining associated items from a large data set is the fact that such mining often generates a huge number of itemsets satisfying the minimum support and minimum confidence threshold. To simplify, we assume min support equals 0.001 and min confidence equals 0.35 and get 5 associated pairs of itemsets. 

The output is as follows.

| antecedents | consequents | antecedent support | consequent support | support  | confidence | confidence_inverse | conviction | conviction_inverse | leverage |
| :---------: | :---------: | :----------------: | :----------------: | :------: | :--------: | :----------------: | :--------: | :----------------: | :------: |
|    21960    |    18151    |      0.003742      |      0.003686      | 0.001797 |   0.4803   |      0.487619      |  1.917096  |      1.94437       | 0.001783 |
|    36992    |    7670     |      0.002612      |      0.002801      | 0.001116 |  0.427419  |      0.398496      |  1.741587  |      1.658158      | 0.001109 |
|    28285    |    12783    |      0.002654      |      0.002808      | 0.001123 |  0.42328   |        0.4         |  1.729076  |      1.662244      | 0.001116 |
|    46201    |    46436    |      0.003096      |      0.002927      | 0.001144 |  0.369615  |      0.390887      |  1.581687  |      1.63665       | 0.001135 |
|    46282    |    46436    |      0.003019      |      0.002927      | 0.001109 |  0.367442  |      0.378897      |  1.576254  |      1.605178      |  0.0011  |

| antecedents | consequents |   lift   | chi_square | p_value | all_confidence | max_confidence | kulczynski |  cosine  | imbalance_ratio |
| :---------: | :---------: | :------: | :--------: | :-----: | :------------: | :------------: | :--------: | :------: | :-------------: |
|    21960    |    18151    | 130.3169 |  33096.4   |    0    |     0.4803     |    0.487619    |  0.48396   | 0.483946 |    0.009975     |
|    36992    |    7670     | 152.5909 |  24075.1   |    0    |    0.398496    |    0.427419    |  0.412958  | 0.412705 |    0.044118     |
|    28285    |    12783    | 150.7354 |  23929.3   |    0    |      0.4       |    0.42328     |  0.41164   | 0.411476 |    0.035599     |
|    46201    |    46436    | 126.2584 |   20378    |    0    |    0.369615    |    0.390887    |  0.380251  | 0.380102 |    0.034532     |
|    46282    |    46436    | 125.5162 |  19633.4   |    0    |    0.367442    |    0.378897    |  0.373169  | 0.373125 |    0.018868     |

Denote $A$ as the antecedent item and $B$ as the consequent item. We calculate the following confidence:
$$
\text{confidence}(A⇒B) = P(B|A) = \frac{\text{support}(A∪B)}{\text{support}(A)} \\

\text{confidence\_inverse}(A⇒B)= \text{confidence}(B⇒A) = \frac{\text{support}(A∪B)}{\text{support}(B)}\\
$$

Apart from confidence, conviction (and inverse) is also derived from the following formulation:
$$
\text{conviction}(A⇒B) = \frac{1 - \text{support}(B)}{1-\text{confidence}(A⇒B)}\ \\
\text{conviction\_invserse}(A⇒B) = \text{conviction}(B⇒A) = \frac{1 - \text{support}(A)}{1-\text{confidence}(B⇒A)}
$$
Conviction measures the dependence of consequent on antecedent $(0, +\infty)$. A relatively high conviction value indicates relatively low support of consequent and relatively high confidence of consequent given the antecedent, and thus indicates high dependence of the consequent on the antecedent.

Leverage is another measure of whether antecedents and consequents are independent, which is derived from the following formulation:
$$
\text{leverage}(A⇒B) = \text{support}(A⇒B) -\text{support}(A) \times \text{support}(B)
$$
Leverage measures how much the support of itemset containing both the consequent and antecedent deviates from the assuming value if $A$ and $B$ are independent.

In addition to the independent/dependent relation between itemsets, we are more interested in measuring how much the antecedent increase the probability of the consequent in transactions. Therefore, we calculate the lift and conduct the $\chi^2$ test with 1 degree of freedom to test independence. If the null hypothesis is rejected, then we can conclude that $A$ and $B$ are statistically not independent.
$$
\text{lift}(A,B) = \frac{\text{support}(A⇒B)}{\text{support}(A)\times\text{support}(B)}
$$

$$
\begin{array}  \ \chi^2_\text{stat} = n \times \left( \frac{(x_3-x_1\times x_2)^2}{x_1\times x_2} + \frac{(x_1-x_3-x_1\times(1-x_2))^2}{x_1\times(1-x_2)} + \frac{(x_2-x_3-x_2\times(1-x_1))^2}{x_2\times(1-x_1)}+ \frac{(1-x_1-x_2+x_3-(1-x_1)\times(1-x_2))^2}{(1-x_1)\times(1-x_2))} \right)\end{array}
$$

where $n$ denotes the number of total transaction, and $x_1, x_2, x_3$ denote the support of antecedent $A$ , consequent $B$ and $A\cup B$.

Generally, lift that are larger than 1 or $p$-value of $\chi^2_{\text{stat}}$ that are smaller than the proposed significance level (e.g. 1%) indicates significant increase of the probability of consequent given the antecedent compared with the original probability of consequent in the whole transactions. However the lift and $\chi^2$ are not null-invariant, which means that the number of transactions that do not contain the antecedent or the consequent will essentially affect the value of these two statistics. When the number of total transaction is significantly larger than the support count of the antecedent and consequent, these statistics will be inflated and therefore become less reliable.

Hence, we turn to null-invariant statistics, including all confidence, max confidence, Kulczynski, and cosine measures. 
$$
\text{all\_confidence}(A,B) = \frac{\text{support}(A∪B)}{\max{(\text{support}(A),\text{support}(B))}} \\
\text{max\_confidence}(A,B) = \max{\text{confidence}(A⇒B),\text{confidence}(B⇒A)} \\
\text{Kulc}(A, B) = \frac{(\text{confidence}(A⇒B)+ \text{confidence}(B⇒A))}{2} \\
\text{cosine}(A,B) = \sqrt{\text{confidence}(A⇒B)×\text{confidence}(B⇒A))}
$$
These four measures are null-invariant since Its value is not influenced by the total number of transactions. However, they are still prone to imbalanced support of antecedent and consequent, which is described by the imbalance ratio.
$$
\text{imbalance\_ratio}=\frac{|\text{support}(A)-\text{support}(B)|}{\text{support}(A)+\text{support}(B)-\text{support}(A∪B)}
$$
When imbalance ratio is relatively large, max_confidence and cosine indicator will be negatively correlated. For such imbalanced support, it is recommended to take into account both imbalanced ratio and Kulc which is considered as a neutral indicator for correlation. Nonetheless, in our case for all correlated items, the imbalance ratio is relatively small and these four indicators are all reliable.

#### 2.2.4 Analysis and Business Insight

| antecedents | consequents | brand_antecedents | brand_type_antecedents | brand_consequents | brand_type_consequents |
| ----------- | ----------- | ----------------- | ---------------------- | ----------------- | ---------------------- |
| 21960       | 18151       | 946               | Established            | 946               | Established            |
| 36992       | 7670        | 1143              | Established            | 1143              | Established            |
| 28285       | 12783       | 946               | Established            | 946               | Established            |
| 46201       | 46436       | 56                | Local                  | 56                | Local                  |
| 46282       | 46436       | 56                | Local                  | 56                | Local                  |

| antecedents | consequents | category_antecedents | category_consequents |
| ----------- | ----------- | -------------------- | -------------------- |
| 21960       | 18151       | Grocery              | Grocery              |
| 36992       | 7670        | Grocery              | Grocery              |
| 28285       | 12783       | Grocery              | Grocery              |
| 46201       | 46436       | Grocery              | Grocery              |
| 46282       | 46436       | Grocery              | Grocery              |

| antecedents | consequents | coupon_id_antecedents | coupon_id_consequents |
| ----------- | ----------- | --------------------- | --------------------- |
| 21960       | 18151       |                       |                       |
| 36992       | 7670        | [932   11 933]        | [932 933 11]          |
| 28285       | 12783       |                       |                       |
| 46201       | 46436       | [ 9 21 30]            | [ 9 21 30]            |
| 46282       | 46436       | [ 9 21 30]            | [ 9 21 30]            |

By merging the correlated items with datasets *item* and *coupon-item mapping*, we find that when two items are considered strongly correlated, they are likely to belong to same brand, same brand type and same category. Furthermore, they tend to be covered by same set of coupons.

Hence, recommendations for itemsets covered by coupons or campaigns (e.g. packaged sales) are as follows:

1. Items from category *Grocery* are more most frequent to be purchased and therefore coupons covering this category are potentially more likely to be redeemed.
2. Items from same brand/brand type are more likely to be bought together and coupons covering correlated products of same brand/brand type are possibly more likely to be redeemed.
3. Items with strong correlation without coupon could be good candidates for future campaigns.

### 2.3 Clustering

#### 2.3.1. Data Pre-Processing

The dataset that used to perform clustering is gained by merging three different datasets, namely 'Customer_demographics.csv', 'Customer_transaction_data.csv' and 'item_data.csv'. 

Specifically, 'Customer_demographics.csv' and 'Customer_transaction_data.csv' are merged together by common feature name 'customer_id'. Similarly, 'item_data.csv' is added to the dataset by 'item_id'.

Then, new features are created by grouping and aggregating at customer level. The newly created features are listed below. Each and every feature is used to describe a customer.

|    Transactions     |      Frequencies       |      Demographics      |
| :-----------------: | :--------------------: | :--------------------: |
|  sum_selling_price  |      visit_times       |      income_level      |
| sum_other_discount  |        Alcohol         |    age_range_18-25     |
| sum_coupon_discount |         Bakery         |    age_range_26-35     |
|   sum_of_quantity   | Dairy, Juices & Snacks |    age_range_36-45     |
|                     |    Flowers & Plants    |    age_range_46-55     |
|                     |          Fuel          |    age_range_56-70     |
|                     |         Garden         |     age_range_70+      |
|                     |        Grocery         | marital_status_Married |
|                     |          Meat          | marital_status_Single  |
|                     |     Miscellaneous      | marital_status_Unknown |
|                     |    Natural Products    |        rented_0        |
|                     |     Packaged Meat      |        rented_1        |
|                     |     Pharmaceutical     |    no_of_children_1    |
|                     |     Prepared Food      |    no_of_children_2    |
|                     |       Restaurant       |   no_of_children_3+    |
|                     |         Salads         | no_of_children_Unknown |
|                     |        Seafood         |     family_size_1      |
|                     |    Skin & Hair Care    |     family_size_2      |
|                     |         Travel         |     family_size_3      |
|                     |    Vegetables (cut)    |     family_size_4      |
|                     |                        |     family_size_5+     |

Specifically, the graph below has shown the process of creating these features mentioned above.

![image](https://user-images.githubusercontent.com/62502750/211182177-d22af452-d521-45a6-b14c-a076ddb9d38f.png)

To summarize, we have created multiple features that can be categorized into three types, namely **Demographics**, **Transaction** and **Frequencies**.

1. **Demographics**

   This dataset includes customer demographic information that is used to divide customers into different groups according to various indicators. These customer information could help the company to craft marketing campaigns which are more closely related with their target consumers and effectively stimulate their needs. Moreover, the deep understanding over the company's potential purchasing force is actually the very practice mirroring the current customer orientation marketing phenomenon, under which the business entities won't thrive unless they consistently improve and sharpen their customer focus to align company's goal with customers goals.

   In this case, we cleaned the data and extract the following typical customer demographics: *Income level, Age range, Marital status, Rent, Number of children* and *Family size*. Specifically, we set *Age range, Marital status, Rent, Number of children* and *Family size* as dummy variables, taking values of 0 or 1 to indicate the absence or presence of the corresponding categories. Take *Age range* as an example, if the very customer's age is between 18-25, then the value of this dummy variable is 1 otherwise it's 0.

2. **Frequencies**

   The *visit_times* is the times that a certain customer visits ABC retailer. For instance, if one entry's *visit times* equals 20, it means that this customer has visited ABC retailer for 20 times throughout the whole record.

   The rest are the consuming frequencies of each kind of products, including Alcohol, Bakery, Daily Juice&Snacks, Flower&Plants, Fuel,Garden, Grocery, Miscellaneous, Natural Products, Packaged Meat, Pharmaceutical, Prepared Food, Restaurant, Salads, Seafood, Skin&Hair care, Travel and Vegetables. For instance, if one entry's *Alcohol* is equal to 10, it means that this customer has bought Alcohol for 10 times throughout the whole record.

3. **Transactions**

   *Sum_selling_price* : Actual amount of money a customer has spent throughout the whole record, i.e. total sales value.

   *Sum_other_discount*: Total discount value spent by a customer. Other discount is from other sources such as manufacturer coupon/loyalty card).

   *Sum_coupon_discount*: Total discount availed from retailer coupon of a customer.

   *Sum_of_quantity*: Total bought quantity of a customer throughout the whole record.

#### 2.3.2 Model selection and development

The objective of K-means is to gaze into the raw data and find interesting/insightful pattern in the mess instead of predicting. In this terminology, the k is the number of clusters you’d like to classify your data into and it should be predefined.

In this case, we choose *the Elbow method* to determine the optimal value of *k* to perform the K-Means clustering algorithm. The basic practice is to perform k means clustering with different values of k, and each of them corresponds with different average distances towards the centroids. The second step is to plot all these points and you can find the elbow point over which the distance from the centroids falls sharply. From the elbow plot, the elbow in this circumstance is 3, which instructs us to divide customers into three groups.

![image](https://user-images.githubusercontent.com/62502750/211182176-b49c5644-8a5f-4167-af43-91dce9f5b7e3.png)

#### 2.3.3. Analysis and Business Insight

After clustering by choosing k=3, we got the 3 group of customers (use cluster 1, 2, 3 to denote respectively) and their attributes are listed as below. The size of each cluster is 1150, 2570 and 339. Next, we will try to interpret the value of each centroids, since it will provide valuable information about the average profile of each cluster.

**General Transaction and amount and visit times**

| **Group** | **Total Transaction Amount** | **Coupon Discount Used** | **Other Discount Used** | **Total Quantity** | **Total Visit Times** |
| :-------: | :--------------------------: | :----------------------: | :---------------------: | :----------------: | :-------------------: |
|     1     |         97075.15098          |       -582.1902448       |      -15823.58252       |    44944.61888     |      92.04370629      |
|     2     |         170887.6757          |       -883.4816779       |      -24073.91859       |    364018.5638     |      133.1543624      |
|     3     |         297275.3451          |       -2066.186923       |       -35376.7941       |    954669.8974     |      183.6923077      |
| **Mean**  |       **121819.7217**        |     **-717.4116842**     |    **-18444.47111**     |  **154183.1763**   |    **104.8065789**    |

![image](https://user-images.githubusercontent.com/62502750/211182196-579d1786-7fc2-424b-81b0-22639bf2bbcd.png)

The table above and bar-plot illustrate that there exists an ascending trend in all these attributes including *Total Transaction Amount , Coupon Discount Used, Other Discount Used, Total Quantity* and *Total Visit Times* from cluster 1 through cluster 3. Cluster 3 has the highest consumption ability (spending power) . And they are also the group visit the supermarket most frequently. They consumed $\$297275.34$ within the data collecting period on average, which is 144.03% higher than the mean level of all the customers in three groups, the quantity they purchased is $154183.18$ on average, 519.18% higher than the mean level. As for the visiting time, they visited the supermarket for averagely 104.81 times, which is also 75.27% above the mean of all the customers.

Also by interpreting the figures above, we can find the trend that there is a large variation between the amount of coupon and other discount, on average, the amount of other discount used is -18444.47, almost constitute 15.14% of the total transaction amount, while the amount of coupon discount used is only 0.59% of the total transaction amount, 717.41 in total. Perhaps by combine the coupon with other promotion activities, can the supermarket receive a better effect.

**Demographic Attributes**

We analyze the demographic attributes of customers in each cluster from the aspects of *Income*, *Age*, *Marital Status*, *Family Size* and *Number of Children*.

**Income**

| **Group** | **Income level** |
| --------- | ---------------- |
| 1         | 4.524476         |
| 2         | 5.214765         |
| 3         | 5.615385         |
| Mean      | 4.715789         |

The income of cluster forms an ascending trend, with cluster 3 having the highest in come level on average, which is also 19.08% higher than the total mean level.

**Age**

| **Group** | **18-25** | **26-35** | **36-45** | **46-55** | **56-70** | **70+** |
| --------- | --------- | --------- | --------- | --------- | --------- | ------- |
| 1         | 6.6%      | 16.8%     | 22.4%     | 36.5%     | 7.9%      | 9.8%    |
| 2         | 4.7%      | 17.4%     | 31.5%     | 31.5%     | 6.7%      | 8.1%    |
| 3         | 0.0%      | 20.5%     | 30.8%     | 38.5%     | 10.3%     | 0.0%    |
| Mean      | 5.9%      | 17.1%     | 24.6%     | 35.7%     | 7.8%      | 8.9%    |

**Marital Status**

| **Group** | Married | Single | Unknown |
| --------- | ------- | ------ | ------- |
| 1         | 39.16%  | 15.73% | 45.10%  |
| 2         | 46.98%  | 14.77% | 38.26%  |
| 3         | 58.97%  | 5.13%  | 35.90%  |
| Mean      | 41.71%  | 15.00% | 43.29%  |

For Age and Marital Status among these groups, average age of each cluster is descending but also close in each group (around 45-47 on average). Group 3 has the most concentrated age attribute, with no customers aged 18-25 or above 70, it mostly consisted of young adults and middle-aged customers. It also has the highest Married rate (58.97%, 17.26pp higher than mean level) and lowest single rate (5.13%, 9.87pp lower than mean level).

**Family Size**

| **Group** | **1**  | **2**  | **3**  | **4**  | **5+** |
| --------- | ------ | ------ | ------ | ------ | ------ |
| 1         | 33.57% | 39.51% | 14.34% | 5.59%  | 6.99%  |
| 2         | 32.89% | 41.61% | 12.08% | 6.04%  | 7.38%  |
| 3         | 17.95% | 38.46% | 10.26% | 17.95% | 15.38% |
| Mean      | 32.63% | 39.87% | 13.68% | 6.32%  | 7.50%  |

**Number of Children**

| **Group** | **1**  | **2**  | **3+** | **Unknown** |
| --------- | ------ | ------ | ------ | ----------- |
| 1         | 14.69% | 6.47%  | 7.52%  | 71.33%      |
| 2         | 12.75% | 7.38%  | 7.38%  | 72.48%      |
| 3         | 10.26% | 17.95% | 15.38% | 56.41%      |
| Mean      | 14.08% | 7.24%  | 7.89%  | 70.79%      |

On average, cluster 3 has a larger family size than the other two clusters, with a nearly 3-people size in family members. Also, the proportion of customers with 4 or more family members is highest in cluster 3, which is almost 2 or 3 times as the proportion in the other 2 clusters.

Also, the same trend is observed in number of children when we take customers who did fill in this question. The dominant number of children within cluster 3 is 3+, whilst customers with children in cluster 1 and 2 mainly have 1 child.

Based on all these attributes taken into consideration, we can make a general profile for each cluster. Cluster 3 is the young adults and middle-aged customers with highest income level, mainly married and having 3 or more children .

3. **Item Purchasing Frequencies**

| **Group** | **Alcohol** | **Bakery** | **Dairy, Juices & Snacks** | **Flowers & Plants** | **Fuel** | **Garden** | **Grocery** | **Meat** | **Miscellaneous** | **Natural Products** |
| :-------: | :---------: | :--------: | :------------------------: | :------------------: | :------: | :--------: | ----------- | -------- | ----------------- | -------------------- |
|   **1**   |  0.854895   |  12.35839  |          15.83392          |       0.956294       | 4.166084 |  0.326923  | 644.6836    | 16.20804 | 1.307692          | 37.08392             |
|   **2**   |  1.080537   |  21.61074  |          23.26846          |       2.114094       | 27.36242 |  0.436242  | 916.8926    | 18.95302 | 4.946309          | 57.78523             |
|   **3**   |  1.128205   |  29.4359   |          23.17949          |       3.461538       | 55.82051 |  1.076923  | 1363.564    | 28.87179 | 12.82051          | 93.51282             |
| **Mean**  |  0.913158   |  15.04868  |          17.66842          |       1.311842       | 11.36447 |  0.386842  | 734.9408    | 17.39605 | 2.611842          | 44.03816             |

| **Group** | **Pharmaceutical** | **Prepared Food** | **Restaurant** | **Salads** | **Seafood** | **Skin & Hair Care** | **Travel** | **Vegetables (cut)** | **Established** | **Local** | **Packaged Meat** |
| :-------: | :----------------: | :---------------: | :------------: | :--------: | :---------: | :------------------: | :--------: | :------------------: | :-------------: | :-------: | :---------------: |
|   **1**   |      105.9126      |     8.798951      |    0.113636    |  0.316434  |  4.620629   |       3.234266       |  0.171329  |       0.08042        |    671.5122     | 230.2517  |     44.73601      |
|   **2**   |      153.7718      |     13.78523      |    0.234899    |  0.194631  |  6.228188   |       6.174497       |  0.154362  |       0.067114       |       948       | 361.5638  |     54.50336      |
|   **3**   |      262.5641      |     23.69231      |    0.74359     |  0.179487  |  13.66667   |       6.25641        |  0.25641   |       0.230769       |    1416.974     | 590.5385  |     87.05128      |
| **Mean**  |      123.3342      |     10.54079      |    0.169737    |  0.285526  |     5.4     |       3.965789       |  0.172368  |       0.085526       |    763.9724     | 274.4842  |     48.82237      |

By analyzing the purchasing frequency for different categories, we found no obvious item preference existing within each cluster, people purchase grocery and established products much more frequent than other products.

The purchasing frequencies pattern is consistent with income pattern, follows an increasing trend from cluster 1 to cluster 3.

Based on the analysis we did above, the customer we should mainly focused is cluster 3, and combined with their attributes and profiles, we should focus more on consumers who is relatively younger, married, having more children and larger family size, also with a higher income and goes to the ABC retailer frequently.

### 2.4. Classification

The above three models only reveal partial information of coupon redemption from the perspective of campaigns, items and customers respectively. Although this could provide a general guideline of the discount marketing strategies, a model to predict accurately whether a customer will redeem a specific coupon of a certain campaign is still required to design more target discount marketing campaigns.

Therefore we combine datasets *train*, *campaign*, *customer transaction* and *item* so as to capture the whole picture of the coupon redemption process/behavior.

#### 2.4.1. Data Pre-Processing

**Standard Process for Feature Engineering**

The interactions between datasets makes it complicated to engineer features since one subject (e.g. campaign/coupon/customer) can be described from multiple perspectives and with data from different datasets. Hence, it is necessary to set up some standard process for feature engineering.

For date features, we decompose it into year, quarter, month, week, weekday, etc. If the observation involves start and end date, a new feature *duration* will be  calculated by using final date subtracting start one, and another feature *recency* will be derived from the difference between the assumed current date (the latest date we have in this dataset) and the last date of the observations.

For any id (e.g. coupon/customer) that corresponds to multiple observations, we design three different strategies for three scenarios:

- For any id corresponding to categorical features with large number of unique levels, we include the most frequent level as well as the Gini index of the relative frequency of each level covered to measure its the distribution.
- For any id corresponding to categorical features with small number of unique levels, we compute the frequency of each level as new features.
- For one id corresponding to numerical features with multiple records, we compute statistics of sum, min, max, mean and standard deviation to represent the distribution.

**Reduce Dimensionality**

After summarizing information from the datasets mentioned and we find that, even without dummy variables, there are 200+ features. With dummy variable, we have 260+ features. This will cause two potential issues:

- Curse of dimensionality. As the number of features increase, after a certain point, the accuracy of a classifier will start dropping. On the one hand, data matrix with high dimensionality is a very sparse matrix. In order to keep the performance at an acceptable level, the training data we need is growing exponentially. What is more, adding too much features will also increase the chance of over-fitting.
- For some computation-intensive algorithms, a large amount of features will result in long training time.

In order to bypass the aforementioned issues, Principle Component Analysis is implemented to compress the dimensionality. PCA uses an orthogonal transformation to linearly transform the observation of a series of potentially correlated variables so as to project them as a series of linearly uncorrelated variables, which are called principal components. Transformation by PCA can preserve most of the variation in dataset while reduce the dimensionality significantly, thus improve the performance of model training.

We implemented the PCA algorithm by using python package scikit-learn. To determine the number of principle components, we conduct grid search to find the optimal value to balance the trade-off between dimensionality reduction and variation retained.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

scaler = StandardScaler()
X_train_ = scaler.fit_transform(X_train)
lst_ratio = []

for n in range(1, 262):
  pca = PCA(n_components=n)
  X_train_ = pca.fit_transform(X_train_)
  lst_ratio.append(pca.explained_variance_ratio_.sum())

y = lst_ratio
x = list(range(1, 262))
plt.figure(dpi=500)
plt.plot(x, y)
plt.xlabel('num of principal components')
plt.ylabel('explained variation ratio')
plt.title('Variation Explained by Principle Components')
plt.show()
```



![image](https://user-images.githubusercontent.com/62502750/211182220-83a78f90-75ad-437e-a11f-dae837cb35dd.png)

At the result, we find that by compressing 261 features into 140 principal components, we can retain 99% variation in the original dataset.

**Oversampling Minor Class**

Only about 0.93% of the training data is labelled as positive (*redemption_status*=1). Therefore, this dataset is extremely unbalanced. Under this case, it is hard for machine learning algorithms to mine information from the minority class/label, thus, leading to poor performance of minority class prediction. Low precision/recall and F1-Score is often observed in this case.

![image](https://user-images.githubusercontent.com/62502750/211182217-012a6ed2-5df1-439f-8bfb-f71b4727ad9f.png)

To address this problem, we used *synthetic minority over-sampling technique (SMOTE)*. *SMOTE* algorithm will first set up the total number of oversampling observations ($N$).  Then, the iteration begins by first selecting a minor class instance at random, and next the K nearest neighbor for that instance is obtained. Finally, $N$ of these $K$ instances is chosen to interpolate new synthetic instances. 

![image](https://user-images.githubusercontent.com/62502750/211182223-81d0d623-2f8a-4f30-9f29-3e0b3080a127.png)

```python
from imblearn.over_sampling import SMOTE
smt = SMOTE()
...
...
  X_train_smt, y_train_smt = smt.fit_resample(X_train_fold, y_train_fold)
  xgb.fit(X_train_smt, y_train_smt.squeeze())
...
...
```

As a result, instability at both algorithm and data level will be inflated since *SMOTE* synthesizes instances, which is similar to some data augmentation tricks in deep learning.

#### 2.4.2 Model Selection and Development

XGBoost algorithm is implemented as the classifier since it is highly efficient, flexible and portable. Kaggle competitions in recent years proved that XGBoost works outstandingly well especially with structured data (even better than deep learning in most cases). XGBoost is a modified algorithm based upon decision trees, which optimized gradient boosting algorithm through parallel processing, tree-pruning and regularization to avoid overfitting and bias.

![image](https://user-images.githubusercontent.com/62502750/211182225-f8154c9d-7d21-4332-8937-d41613b24f88.png)

We choose the number of weak learners (decision trees) `n_estimator=100`. If the number of estimator is too large, it is prone to over-fit the data. If it is too small, it is prone to under-fit the data. Therefore, choosing a moderate number of estimator is crucial. In real practice, best hyper parameters is usually determined by using Grid Search, which, due to time limit, is not included in this part.  Similarly, a moderate learning rate of *0.08* is chosen.

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import xgboost

skf = StratifiedKFold(n_splits=5, shuffle=True)
smt = SMOTE()
xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

lst_acc_train = []
lst_acc_test = []


i=0
for train_index, test_index in skf.split(X_train, y_train):
  i += 1
  X_train_fold, X_test_fold = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
  y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

  X_train_smt, y_train_smt = smt.fit_resample(X_train_fold, y_train_fold)
  xgb.fit(X_train_smt, y_train_smt.squeeze())
  y_pred_train = xgb.predict(X_train_fold)
  y_pred_test = xgb.predict(X_test_fold)

  lst_acc_train.append(accuracy_score(y_train_fold, y_pred_train))
  lst_acc_test.append(accuracy_score(y_test_fold, y_pred_test))

  print('-'*20+f'{i} iteration'+'-'*20)
  print('XGB accuracy score for train: %.3f and test: %.3f' % (
        accuracy_score(y_train_fold, y_pred_train),
        accuracy_score(y_test_fold, y_pred_test)))
  print('Classification report for train:')
  print(classification_report(y_train_fold, y_pred_train))
  print('Classification report for test:')
  print(classification_report(y_test_fold, y_pred_test))
  print()
```

#### 2.4.3 Model Evaluation

We decided to use Stratified K-Fold cross validation method to evaluate the performance of our data, since there is no assigned test set with labels.

![image](https://user-images.githubusercontent.com/62502750/211182231-74cadb9d-8807-48d3-8573-131fc25ae351.png)

From the result above, we can see that the model performed and generalize well overall. On average, it has achieved an accuracy of **0.9979**. For the test set, the model has achieved a $F1$-Score of **1** on all iteration with respect to the prediction of negative sample (*redemption_status=0*) and a $F1$-Score of **0.862** on all iteration with respect to the prediction of positive sample (*redemption_status=1*).

## 6. Summary

In conclusion, the model we built by implementing XGBoost can be used to predict the redemption status of coupon for ABC retailer, based upon information of past discount campaigns, customers demographics and transaction items.

Besides, the three main models we have built, including regression, frequent mining and clustering, help to attract business insights and provide a general guideline of discount marketing strategies for ABC company. 

1. ###### **Business recommendation**

   In terms of model results generated based upon past relevant data, we offer the following recommendations for ABC retailer's consideration:

   1. Advice on campaign design (from regression model):

      a. Type X campaigns generally outperform that of type Y.

      b. Shorter campaign duration can potentially leads to higher coupon redemption rate.

   2. Advice on products selection for coupon design (from frequent pattern mining):

      a. Items from category *Grocery* are more most frequent to be purchased and therefore coupons covering this category are potentially more likely to be redeemed.

      b. Items from same brand/brand type are more likely to be bought together and coupons covering correlated products of same brand/brand type are possibly more likely to be redeemed.

   3. Advice on distribution to target customers (from clustering model) :

      Customer who is relatively younger with high income, married and has a big family with more children and visits ABC retailer more frequently should be the main target customers.

2. ###### **Limitations**

   Although quite a large amount data available, discrepancy and inconsistency exit in these 6 datasets, which may distort the result. Besides, some ill-defined levels of categorical features need further clarify by ABC refilter to achieve more accurate outcomes.

   Also, the feasibility of the proposal needs to be verified by experts and more algorithms can be explored and compared before delivering the proposal to ABC retailer.

## 7. Appendix

### 7.1 Regression Source Code

```python
import numpy as np
import pandas as pd

train = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/train.csv")
transaction = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/customer_transaction_data.csv", parse_dates=['date'])

d1 = train.groupby('campaign_id').agg(redemption_frequency=('redemption_status','sum'))
d2 = train['campaign_id'].value_counts().sort_index().rename('total_amount').to_frame()
d2.index.name = d1.index.name
d3 = train.groupby('campaign_id')['coupon_id'].nunique()
coupon_types = train.coupon_id.nunique()
d3 = (d3 / coupon_types * 100).rename('coupon_covering_ratio%')
d4 = train.groupby('campaign_id')['customer_id'].nunique()
n_customers = train.customer_id.nunique()
d4 = (d4 / n_customers * 100).rename('customer_covering_ratio%')

d = pd.concat([d1, d2, d3, d4], axis=1)
d['redemption_ratio%'] = d['redemption_frequency'] / d['total_amount'] * 100
d = d.drop(columns=['redemption_frequency'])

for index, row in campaign.iterrows():
  if row.end_date < row.start_date:
    campaign.iloc[index, 2] = row.end_date
    campaign.iloc[index, 3] = row.start_date

    campaign['duration_days'] = (campaign['end_date'] - campaign['start_date']).dt.days

campaign['start_year'] = campaign['start_date'].dt.year
campaign['start_quarter'] = campaign.start_date.dt.quarter
campaign['start_week'] = campaign.start_date.dt.isocalendar().week
campaign['start_weekday'] = campaign.start_date.dt.weekday

campaign['end_year'] = campaign['end_date'].dt.year
campaign['end_quarter'] =  campaign.end_date.dt.quarter
campaign['end_week'] =  campaign.end_date.dt.isocalendar().week
campaign['end_weekday'] =  campaign.end_date.dt.weekday
df = campaign.drop(columns=['start_date', 'end_date'])
df = df.merge(d, on='campaign_id', how='right')

df['amount/day'] = df['total_amount'] / df['duration_days']
df["redemption_ratio"] = df['redemption_ratio%']
df = df.drop(columns = ['redemption_ratio%'])
df.rename(columns={"redemption_ratio":'redemption_ratio%'}, inplace=True)

df.to_csv('/content/drive/MyDrive/Project Data Mining/regression_campaign.csv')
```

### 7.2 Frequent Pattern Mining Source Code

```python
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from scipy import stats

def chi2_pvalue(x):
    '''
    Return the p-value of given chi-square(dof=1) statistics. 
    '''
    return 1 - stats.chi2.cdf(x, 1)

def chi_square(df:pd.DataFrame, transaction_num:int):
    '''
    Input the association rules DataFrame and the number of total transactions.
    Return a DataFrame with two new columns indicating the chi square statistic, corresponding p-value (dof=1).
    '''
    x1 = df["antecedent support"]
    x2 = df["consequent support"]
    x3 = df["support"]
    df['chi_square'] = np.round(transaction_num * ((x3-x1*x2)**2/(x1*x2) + (x1-x3-x1*(1-x2))**2/(x1*(1-x2)) + (x2-x3-x2*(1-x1))**2/(x2*(1-x1)) + (1-x1-x2+x3-(1-x1)*(1-x2))**2/((1-x1)*(1-x2))), decimals=1)
    df['p_value'] = np.round(df['chi_square'].apply(chi2_pvalue), decimals=3)
    df['leverage'] = x3 - x2*x1

  
def confidences(df:pd.DataFrame):
    '''
    Input the association rules DataFrame.
    Return a DataFrame with new columns of confidences.
    '''
    x1 = df["antecedent support"]
    x2 = df["consequent support"]
    x3 = df["support"]
    df['confidence'] = x3/x1
    df['confidence_inverse'] = x3/x2
    df['lift'] = x3/(x1*x2)
    df['max_confidence'] = x3/np.minimum(x1, x2)
    df['all_confidence'] = x3/np.maximum(x1, x2)
    df['kulczynski'] = (x3/x1 + x3/x2)/2
    df['cosine'] = x3/np.sqrt(x1*x2)

def conviction(df:pd.DataFrame):
    '''
    Input the association rules DataFrame.
    Return a DataFrame with new column of conviction of the original and inverse association.
    '''
    x1 = df["antecedent support"]
    x2 = df["consequent support"]
    x3 = df["confidence"]
    x4 = df["confidence_inverse"]

    df['conviction'] = (1 - x2)/(1 - x3)
    df['conviction_inverse'] = (1 - x1)/(1 - x4)

def imbalance_ratio(df:pd.DataFrame):
    '''
    Input the association rules DataFrame.
    Return a DataFrame with new column of imbalance_ratio.
    '''
    x1 = df["antecedent support"]
    x2 = df["consequent support"]
    x3 = df["support"]
    df['imbalance_ratio'] = np.abs(x1 - x2)/(x1 + x2 - x3)
    
def drop_duplicate_association(df:pd.DataFrame):
    '''
    Input the association rules DataFrame.
    Drop rows of duplicate assocaitions.
    '''
    df['association'] = [ tuple(sorted(list(x|y))) for x,y in zip(df["antecedents"], df["consequents"])]
    df.drop_duplicates(subset=['association'], inplace=True)
    df.drop(columns=['association'], inplace = True)
    df.index=range(0, len(df))

    
def frequent_pattern(df, min_support:float, apriori=False):
  '''
  Input a transactions series and minimal support.
  Return a frequent pattern DataFrame default by fptree algorithm. 
  Apriori algorithm is implemented if apriori=True. 

  '''
  te = TransactionEncoder()
  te_array = te.fit(df).transform(df)
  d = pd.DataFrame(te_array, columns=te.columns_)
  if apriori:
    fp = apriori(d, min_support=min_support, use_colnames=True)
  else:
    fp = fpgrowth(d, min_support=min_support, use_colnames=True)
  return fp

def association(df, min_support:float, min_confidence:float):
  '''
  Input a transactions series, minimal support and minimal confidence.
  Return a association rules DataFrame.
  '''
  n=len(df)
  ar = association_rules(frequent_pattern(df, min_support), metric='confidence', min_threshold=min_confidence).drop(columns=["lift","leverage"])
  chi_square(ar, n)
  confidences(ar)
  conviction(ar)
  imbalance_ratio(ar)
  ar = ar.loc[:,["antecedents", "consequents", "antecedent support", \
                                                 "consequent support", "support", "confidence", "confidence_inverse", \
                                                 "conviction","conviction_inverse", "lift", "chi_square", "p_value", "leverage", "all_confidence", \
                                                                                       "max_confidence", "kulczynski", "cosine", "imbalance_ratio"]]
  return ar

def correlation(df, min_support:float, min_confidence:float):
  '''
  Input a transactions series, minimal support and minimal confidence.
  Return a correlation DataFrame.
  '''
  n = len(df)
  corr = association(df, min_support, min_confidence)
  corr = corr.loc[corr.all_confidence>=min_confidence, :]
  drop_duplicate_association(corr)

  return corr
```

```python
transaction = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/customer_transaction_data.csv", parse_dates=['date'])
fp_item = transaction.groupby(['date', 'customer_id'])['item_id'].unique().reset_index(drop=True)

n_fp = []
n_1itemset = []
for i in range(1, 11):
  min_supp = i/1000
  fp = frequent_pattern(fp_item, min_supp)
  fp_1item = fp[fp.itemsets.apply(len)==1]
  n_fp.append(len(fp))
  n_1itemset.append(len(fp_1item))

min_support = np.array(range(1, 11))/1000
df = pd.DataFrame({'min_support':min_support, "number of frequent patterns":n_fp, "number of frequent 1-itemset":n_1itemset})
df["number of frequent k-itemset(k>1)"]=df.iloc[:,1]-df.iloc[:,2]
print(df)

obs = []
for i in (0.001, 0.002):
  for j in range(20, 65, 5):
    min_confidence = j/100
    n_ar = len(association(fp_item, i, min_confidence))
    n_corr = len(correlation(fp_item, i, min_confidence))
    obs.append({"min_support":i, "min_confidence":min_confidence, "number of association rules":n_ar, "number of correlations":n_corr})

df2 = pd.DataFrame(obs)
print(df2)

item = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/item_data.csv")
coupon_item = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/coupon_item_mapping.csv")

min_support = 0.001
min_confidence = 0.35

corr = correlation(fp_item, min_support, min_confidence)

def value(x:frozenset):
  y, = x
  return y

corr.antecedents = corr.antecedents.apply(value)
corr.consequents = corr.consequents.apply(value)

corr_item = corr.merge(item, left_on='antecedents', right_on='item_id', how='left').merge(item, left_on='consequents', right_on='item_id', suffixes=("_antecedents", "_consequents"),  how='left')

coupon_item_ = coupon_item.groupby('item_id')['coupon_id'].unique()
coup_item_ = coupon_item_.reset_index()
corr_coup = corr_item.merge(coupon_item_, left_on='antecedents', right_on='item_id', how='left').merge(coupon_item_, left_on='consequents', right_on='item_id', suffixes=("_antecedents", "_consequents"),  how='left').drop(columns=['item_id_antecedents','item_id_consequents'])

corr_coupon.to_csv('/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/ccorr_coupon.csv', index=False)
```

### 7.3 Clustering Source Code

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

transactions = pd.read_csv('customer_transaction_data.csv').groupby(['date','customer_id','item_id'])[['quantity','selling_price','other_discount','coupon_discount']].sum().reset_index()
transaction_freq=transactions.groupby(['customer_id','date'])\
.agg(average_selling_price =('selling_price','mean'),average_other_discount = ('other_discount','mean'),\
     average_coupon_discount = ('coupon_discount','mean'),sum_of_quantity = ('quantity','sum')).reset_index()

transaction_customer=transactions.groupby(['customer_id'])\
.agg(sum_selling_price =('selling_price','sum'),sum_other_discount = ('other_discount','sum'),\
     sum_coupon_discount = ('coupon_discount','sum'),sum_of_quantity = ('quantity','sum')).reset_index()

freq = transaction_freq.groupby('customer_id').agg(visit_times = ('date','count'))

transaction_customer = pd.merge(transaction_customer,freq,on='customer_id',how='inner')

# Get dummies

demographic_new = pd.read_csv('customer_demographics.csv')
demographic_new.marital_status = demographic_new.marital_status.fillna('Unknown')
demographic_new.no_of_children = demographic_new.no_of_children.fillna('Unknown')
demographic_new = pd.get_dummies(demographic_new,columns=['age_range','marital_status','rented','no_of_children','family_size'])
transaction_customer_1 = pd.merge(transaction_customer,demographic_new,how='inner',on=['customer_id'])
transaction_customer_1

item = pd.read_csv('item_data.csv').drop(['brand'],axis=1)
'''
encoder = OrdinalEncoder()
item[['brand_type','category']] = encoder.fit_transform(item[['brand_type','category']])
'''

tx_item = pd.merge(transactions,item,on='item_id',how='inner')
tx_item
pivot1 = tx_item.groupby(['customer_id','category']).agg(count=('item_id','count'))\
.pivot_table(index=['customer_id'], columns=['category'], values=['count'])
pivot1.columns=pivot1.columns.droplevel(0)
pivot1 = pivot1.fillna(0)

pivot2 = tx_item.groupby(['customer_id','brand_type']).agg(count=('item_id','count'))\
.pivot_table(index=['customer_id'], columns=['brand_type'], values=['count'])
pivot2.columns=pivot2.columns.droplevel(0)
pivot2 = pivot2.fillna(0)

# join back to df.
customer_clustering = pd.merge(transaction_customer_1,pivot1,on='customer_id',how='inner')
customer_clustering = pd.merge(customer_clustering,pivot2,on='customer_id',how='inner')
customer_clustering = customer_clustering.drop(['customer_id'],axis=1)
from sklearn.cluster import KMeans 

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(customer_clustering)
    inertia.append(kmeans.inertia_)
inertia

plt.plot([x for x in range(1,11)],inertia)
plt.title('Elbow Plot')
plt.ylabel('Sum of Squared Distance')
plt.xlabel('Number of Clusters')
kmeans = KMeans(n_clusters=3)
kmeans.fit(customer_clustering)
customer_mean = customer_clustering.mean(axis=0)
centroiods_3 = pd.DataFrame(kmeans.cluster_centers_)
```



### 7.4 Classification Source Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/train.csv")
test = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/test.csv")
demographics = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/customer_demographics.csv")
transaction = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/customer_transaction_data.csv", parse_dates=['date'])
coupon_item = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/coupon_item_mapping.csv")
item = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/item_data.csv")
campaign = pd.read_csv("/content/drive/MyDrive/Project Data Mining/Predicting Coupon Redemption/campaign_data.csv", parse_dates=['start_date','end_date'])

for index, row in campaign.iterrows():
  if row.end_date < row.start_date:
    campaign.iloc[index, 2] = row.end_date
    campaign.iloc[index, 3] = row.start_date

campaign['duration_days'] = (campaign['end_date'] - campaign['start_date']).dt.days

campaign['start_year'] = campaign['start_date'].dt.year
campaign['start_quarter'] = campaign.start_date.dt.quarter
campaign['start_month'] = campaign.start_date.dt.month
campaign['start_week'] = campaign.start_date.dt.isocalendar().week
campaign['start_day'] = campaign.start_date.dt.day
campaign['start_weekday'] = campaign.start_date.dt.weekday

campaign['end_year'] = campaign['end_date'].dt.year
campaign['end_quarter'] =  campaign.end_date.dt.quarter
campaign['end_month'] = campaign.end_date.dt.month
campaign['end_week'] =  campaign.end_date.dt.isocalendar().week
campaign['end_day'] = campaign.end_date.dt.day
campaign['end_weekday'] =  campaign.end_date.dt.weekday

camp = campaign.drop(columns=['start_date', 'end_date'])

for col in camp.columns[2:]:
  if col == 'start_year' or col == 'end_year':
    camp[f'camp_{col}'] = camp[col].astype('category')
  else:
    camp[f'camp_{col}'] = camp[col]
  camp.drop(columns=col, inplace=True)

camp.campaign_type = camp.campaign_type.astype('category')
```

```python
coupon_item_ = coupon_item.merge(item, on='item_id', how='left').dropna()
d = coupon_item_.groupby('coupon_id')['brand'].value_counts().rename('brand_count').reset_index()
d = d.merge(d.groupby('coupon_id').agg(sum=('brand_count','sum')).reset_index(), on='coupon_id', how='left')
d['relative_frequency'] = d['brand_count'] / d['sum']

ind = d.groupby('coupon_id')['relative_frequency'].nlargest(1).reset_index(level=0).index
mf = d.iloc[ind,:].drop(columns=['brand_count','sum'])

def gini_index(x):
  return 1- sum(np.square(x))
gini = d.groupby('coupon_id').agg(coup_brand_gini=('relative_frequency', gini_index)).reset_index()

coup_1 = coupon_item_.groupby('coupon_id').agg(coup_nitem=('item_id','nunique'), coup_nbrand=('brand','nunique'))
coup_1 = coup_1.reset_index().merge(mf, on='coupon_id', validate='one_to_one')
coup_1 = coup_1.rename(columns={'brand':'coup_mfbrand','relative_frequency':'coup_mfbrand_rf'})
coup_1 = coup_1.merge(gini, on='coupon_id', validate='one_to_one')
coup_1 = coup_1.merge(item.iloc[:, 1:3].drop_duplicates(), left_on='coup_mfbrand', right_on='brand', how='left').drop(columns=['brand', 'coup_mfbrand']).rename(columns={'brand_type':'coup_mfbrand_type'})
coup_1['coup_mfbrand_type'] = coup_1['coup_mfbrand_type'].astype('category')

btype = coupon_item_.groupby('coupon_id')['brand_type'].value_counts().rename('brand_type_count').reset_index()
btype = btype.pivot_table(index=['coupon_id'], columns=['brand_type'], values=['brand_type_count'])
btype = btype.fillna(0)

cate = coupon_item_.groupby('coupon_id')['category'].value_counts().rename('category_count').reset_index()
cate = cate.pivot_table(index=['coupon_id'], columns=['category'], values=['category_count'])
cate = cate.dropna(axis=1, how='all').fillna(0)

bc = btype.merge(cate, on='coupon_id', validate='one_to_one')

for col in btype.columns:
  bc[f'coup_brand_{col[1]}'] = bc[col]
  bc = bc.drop(columns=col)

for col in cate.columns:
  if col[1] == 'Vegetables (cut)':
    s = 'cutVegetables'
  else:
    s = col[1]
    s = s.replace(" ","").replace(',', '').replace('&','')
  
  bc[f'coup_{s}'] = bc[col]
  bc = bc.drop(columns=col)

bc.columns = bc.columns.droplevel(1)

coup_1 = coup_1.merge(bc, on='coupon_id', validate='one_to_one')


trans = transaction.groupby(['date','customer_id','item_id'])[['quantity','selling_price','other_discount','coupon_discount']].sum().reset_index()
trans['original_selling_price'] = trans['selling_price'] - trans['other_discount'] - trans['coupon_discount']
trans['original_unit_price'] = trans['original_selling_price'] / trans['quantity']

items = trans.set_index(['date','customer_id'])['item_id'].value_counts().rename('item_ntransaction')
items.index.name = 'item_id'
items = items.reset_index()

now = trans.date.max()
items = items.merge(trans.groupby('item_id')['date'].max().apply(lambda x: now-x).dt.days.rename('item_recency_days'), on='item_id', validate='one_to_one')
items = items.merge((trans.groupby('item_id')['date'].max()-trans.groupby('item_id')['date'].min()).dt.days.rename('item_life_days'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['customer_id'].nunique().rename('item_popularity'), on='item_id', validate='one_to_one')

items = items.merge(trans.groupby('item_id')['quantity'].sum().rename('item_tot_quantity'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['quantity'].mean().rename('item_avg_quantity'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['quantity'].min().rename('item_min_quantity'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['quantity'].max().rename('item_max_quantity'), on='item_id', validate='one_to_one')

items = items.merge(trans.groupby('item_id')['selling_price'].sum().rename('item_tot_sales_value'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['selling_price'].mean().rename('item_avg_sales_value'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['selling_price'].min().rename('item_min_sales_value'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['selling_price'].max().rename('item_max_sales_value'), on='item_id', validate='one_to_one')

items = items.merge(trans.groupby('item_id')['other_discount'].sum().rename('item_tot_other_discount'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['other_discount'].mean().rename('item_avg_other_discount'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['other_discount'].min().rename('item_min_other_discount'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['other_discount'].max().rename('item_max_other_discount'), on='item_id', validate='one_to_one')

items = items.merge(trans.groupby('item_id')['coupon_discount'].sum().rename('item_tot_coupon_discount'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['coupon_discount'].mean().rename('item_avg_coupon_discount'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['coupon_discount'].min().rename('item_min_coupon_discount'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['coupon_discount'].max().rename('item_max_coupon_discount'), on='item_id', validate='one_to_one')

items = items.merge(trans.groupby('item_id')['original_unit_price'].mean().rename('item_avg_unit_price'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['original_unit_price'].min().rename('item_min_unit_price'), on='item_id', validate='one_to_one')
items = items.merge(trans.groupby('item_id')['original_unit_price'].max().rename('item_max_unit_price'), on='item_id', validate='one_to_one')

coup_items = coupon_item.merge(items, on='item_id', how='left').dropna()
coup_2 = coupon_item['coupon_id'].drop_duplicates().sort_values()
coup_2 = coup_2.reset_index(drop=True).to_frame()

for col in coup_items.columns[2:]:
  for func in ['min', 'max', 'mean', 'std']:
    coup_2 = coup_2.merge(coup_items.groupby('coupon_id').agg({col:func}).rename(columns={col:f'coup_{col}_{func}'}), on='coupon_id', validate='one_to_one')

coup_2 = coup_2.fillna(0)

coup = coup_1.merge(coup_2, on='coupon_id', validate='one_to_one')
```

```python
trans = trans.rename(columns={'selling_price':'expense','original_selling_price':'original_expense','original_unit_price':'original_expense_peritem'})
trans['expense_peritem'] = trans['expense'] /trans['quantity']
trans['other_discount_peritem'] = trans['other_discount'] /trans['quantity']
trans['coupon_discount_peritem'] = trans['coupon_discount'] /trans['quantity']

trans_item = trans.merge(item, on='item_id', how='left')

d = trans_item.groupby('customer_id')['item_id'].value_counts().rename('frequency').reset_index()
d = d.merge(d.groupby('customer_id')['frequency'].sum().rename('total_frequency'), on='customer_id', how='left')
d['relative_frequency'] = d['frequency'] / d['total_frequency']

cust = trans.customer_id.drop_duplicates()
cust = cust.sort_values().reset_index(drop=True).to_frame()
cust = cust.merge(trans.groupby('customer_id')['date'].nunique().rename('cust_nvisit'), on='customer_id', validate='one_to_one')
cust = cust.merge((trans.groupby('customer_id')['date'].max()-trans.groupby('customer_id')['date'].min()).dt.days.rename('cust_span_days'), on='customer_id', validate='one_to_one')
cust = cust.merge(trans.groupby('customer_id')['date'].max().apply(lambda x: (now-x)).dt.days.rename('cust_recency_days'), on='customer_id', validate='one_to_one')
cust = cust.merge(trans.groupby('customer_id')['item_id'].nunique().rename('cust_item_variety'), on='customer_id', validate='one_to_one')

ind = d.groupby('customer_id')['relative_frequency'].nlargest(1).reset_index(0).index
cust = cust.merge(d.loc[ind, ['customer_id','item_id','relative_frequency']].reset_index(drop=True), on='customer_id', validate='one_to_one')
cust = cust.rename(columns={'item_id':'cust_mfitem', 'relative_frequency':'cust_mfitem_rf'})
cust = cust.merge(d.groupby('customer_id').agg(cust_item_gini=('relative_frequency', gini_index)), on='customer_id', validate='one_to_one')

cmf = cust.merge(item, left_on='cust_mfitem', right_on='item_id', how='left')
cust[f'cust_mfitem_category'] = cmf['category'].astype('category')
cust = cust.drop(columns='cust_mfitem')

b = trans_item.groupby('customer_id')['brand'].value_counts().rename('frequency').reset_index()
b = b.merge(b.groupby('customer_id')['frequency'].sum().rename('total_frequency'), on='customer_id', how='left')
b['relative_frequency'] = b['frequency']/b['total_frequency']

cust = cust.merge(trans_item.groupby('customer_id')['brand'].nunique().rename('cust_brand_variety'), on='customer_id',validate='one_to_one')
ind = b.groupby('customer_id')['relative_frequency'].nlargest(1).reset_index(0).index
cust = cust.merge(b.loc[ind, ['customer_id','brand','relative_frequency']].reset_index(drop=True), on='customer_id', validate='one_to_one')
cust = cust.rename(columns={'brand':'cust_mfbrand', 'relative_frequency':'cust_mfbrand_rf'})
cust['cust_mfbrand'] = cust['cust_mfbrand'].astype('category')

cust = cust.merge(b.groupby('customer_id').agg(cust_brand_gini=('relative_frequency', gini_index)), on='customer_id', validate='one_to_one')

df = trans.groupby(['customer_id','date'])[["expense","other_discount","coupon_discount","original_expense"]].sum().reset_index()
for col in df.columns[2:]:
  for func in ['sum', 'max', 'min', 'mean', 'std']:
    if func == 'sum':
      cust = cust.merge(df.groupby('customer_id')[col].sum().rename(f'cust_tot_{col}'), on='customer_id', validate='one_to_one')
    else:
      cust = cust.merge(df.groupby('customer_id').agg({col:func}).rename(columns={col:f'cust_{func}_{col}_pervisit'}), on='customer_id', validate='one_to_one')

for col in trans.columns[-4:]:
   for func in ['max', 'min', 'mean', 'std']:
     cust = cust.merge(trans.groupby('customer_id').agg({col:func}).rename(columns={col:f'cust_{func}_{col}'}), on='customer_id', validate='one_to_one')
        
btype = trans_item.groupby(['customer_id','brand_type']).agg(brand_type_quantity=('quantity','sum'), brand_type_expense=('expense','sum'))
btype = btype.pivot_table(index=['customer_id'], columns='brand_type', values=['brand_type_quantity','brand_type_expense'])
btype.columns = ['cust_Local_expense', 'cust_Established_expense','cust_Local_quantity', 'cust_Established_quantity']

cate = trans_item.groupby(['customer_id','category']).agg(category_quantity=('quantity','sum'), category_expense=('expense','sum'))
cate = cate.pivot_table(index=['customer_id'], columns='category', values=['category_quantity','category_expense'])

for col in cate.columns:
  if col[1] == 'Vegetables (cut)':
    s = 'cutVegetables'
  else:
    s = col[1]
    s = s.replace(" ","").replace(',', '').replace('&','')
  
  if col[0] == "category_expense":
    cate[f'cust_{s}_expense'] = cate[col]
  else:
    cate[f'cust_{s}_quantity'] = cate[col]
  cate = cate.drop(columns=col)

cate = cate.fillna(0)
cate.columns = cate.columns.droplevel(1)

cust = cust.merge(btype, on='customer_id', validate='one_to_one').merge(cate, on='customer_id', validate='one_to_one')
```

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def category2dummy(df_:pd.DataFrame):
  for col in df_.select_dtypes(include=['category']).columns:
    dum = pd.get_dummies(df_[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=False)
    df_ = pd.concat([df_, dum], axis=1)
    df_ = df_.drop(columns=col) 
  return df_

camp_ = category2dummy(camp)
coup_ = category2dummy(coup)
cust_ = category2dummy(cust)

x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_train = x_train.merge(camp_, on='campaign_id', how='left').merge(coup_, on='coupon_id', how='left').merge(cust_, on='customer_id', how='left')\
                                                                               .drop(columns=["id","campaign_id","coupon_id","customer_id"])

scaler = StandardScaler()
X_train_ = scaler.fit_transform(X_train)
lst_ratio = []

for n in range(1, 262):
  pca = PCA(n_components=n)
  X_train_ = pca.fit_transform(X_train_)
  lst_ratio.append(pca.explained_variance_ratio_.sum())

y = lst_ratio
x = list(range(1, 262))
plt.figure(dpi=500)
plt.plot(x, y)
plt.xlabel('num of principal components')
plt.ylabel('explained variation ratio')
plt.title('Variation Explained by Principle Components')
plt.show()
```

```python
!pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import xgboost

scaler = StandardScaler()
pca = PCA(n_components=140)
skf = StratifiedKFold(n_splits=5, shuffle=True)
smt = SMOTE()
xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

lst_acc_train = []
lst_acc_test = []

X_train_scale = scaler.fit_transform(X_train)
X_train = pca.fit_transfor(X_train_scale)

i=0
for train_index, test_index in skf.split(X_train, y_train):
  i += 1
  X_train_fold, X_test_fold = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
  y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

  X_train_smt, y_train_smt = smt.fit_resample(X_train_fold, y_train_fold)
  xgb.fit(X_train_smt, y_train_smt.squeeze())
  y_pred_train = xgb.predict(X_train_fold)
  y_pred_test = xgb.predict(X_test_fold)

  lst_acc_train.append(accuracy_score(y_train_fold, y_pred_train))
  lst_acc_test.append(accuracy_score(y_test_fold, y_pred_test))

  print('-'*20+f'{i} iteration'+'-'*20)
  print('XGB accuracy score for train: %.3f and test: %.3f' % (
        accuracy_score(y_train_fold, y_pred_train),
        accuracy_score(y_test_fold, y_pred_test)))
  print('Classification report for train:')
  print(classification_report(y_train_fold, y_pred_train))
  print('Classification report for test:')
  print(classification_report(y_test_fold, y_pred_test))
  print()
```
