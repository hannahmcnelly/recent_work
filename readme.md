# Data Wrangling Final - Sleep Analysis on Mammals
Hannah McNelly

## Loading in packages

``` python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

## Loading in data

``` python
data = pd.read_csv("~/Downloads/dataset_2191_sleep.csv.xls")
```

## Looking into the data

``` python
data.head()
data.tail()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | body_weight | brain_weight | max_life_span | gestation_time | predation_index | sleep_exposure_index | danger_index | total_sleep |
|----|----|----|----|----|----|----|----|----|
| 57 | 2.000 | 12.3 | 7.5 | 200 | 3 | 1 | 3 | 5.4 |
| 58 | 0.104 | 2.5 | 2.3 | 46 | 3 | 2 | 2 | 15.8 |
| 59 | 4.190 | 58.0 | 24 | 210 | 4 | 3 | 4 | 10.3 |
| 60 | 3.500 | 3.9 | 3 | 14 | 2 | 1 | 1 | 19.4 |
| 61 | 4.050 | 17.0 | 13 | 38 | 3 | 1 | 1 | ? |

</div>

``` python
data.dtypes
```

    body_weight             float64
    brain_weight            float64
    max_life_span            object
    gestation_time           object
    predation_index           int64
    sleep_exposure_index      int64
    danger_index              int64
    total_sleep              object
    dtype: object

``` python
data.describe(include= 'all')
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | body_weight | brain_weight | max_life_span | gestation_time | predation_index | sleep_exposure_index | danger_index | total_sleep |
|----|----|----|----|----|----|----|----|----|
| count | 62.000000 | 62.000000 | 62 | 62 | 62.000000 | 62.000000 | 62.000000 | 62 |
| unique | NaN | NaN | 48 | 50 | NaN | NaN | NaN | 45 |
| top | NaN | NaN | ? | ? | NaN | NaN | NaN | ? |
| freq | NaN | NaN | 4 | 4 | NaN | NaN | NaN | 4 |
| mean | 198.789984 | 283.134194 | NaN | NaN | 2.870968 | 2.419355 | 2.612903 | NaN |
| std | 899.158011 | 930.278942 | NaN | NaN | 1.476414 | 1.604792 | 1.441252 | NaN |
| min | 0.005000 | 0.140000 | NaN | NaN | 1.000000 | 1.000000 | 1.000000 | NaN |
| 25% | 0.600000 | 4.250000 | NaN | NaN | 2.000000 | 1.000000 | 1.000000 | NaN |
| 50% | 3.342500 | 17.250000 | NaN | NaN | 3.000000 | 2.000000 | 2.000000 | NaN |
| 75% | 48.202500 | 166.000000 | NaN | NaN | 4.000000 | 4.000000 | 4.000000 | NaN |
| max | 6654.000000 | 5712.000000 | NaN | NaN | 5.000000 | 5.000000 | 5.000000 | NaN |

</div>

## Cleaning the data

``` python
data.replace(['?'], np.nan, inplace=True)
```

``` python
data.isnull().sum()
```

    body_weight             0
    brain_weight            0
    max_life_span           4
    gestation_time          4
    predation_index         0
    sleep_exposure_index    0
    danger_index            0
    total_sleep             4
    dtype: int64

``` python
cols_fixed = ['max_life_span', 'gestation_time', 'total_sleep']

for col in cols_fixed:
    data[col] = pd.to_numeric(data[col], errors='coerce')
```

``` python
data = data.dropna(subset=cols_fixed)
```

``` python
data.dtypes
```

    body_weight             float64
    brain_weight            float64
    max_life_span           float64
    gestation_time          float64
    predation_index           int64
    sleep_exposure_index      int64
    danger_index              int64
    total_sleep             float64
    dtype: object

I wrote this code to prepare the data to best answer the 3 questions I
had below. To start I replaced any values that were inconsistent to
display NaN. I also converted non-numeric values to numeric to be able
to work with the data to create visualizations.

## Question 1:

# How spread out is the total hours of sleep for mammals?

``` python
plt.figure(figsize=(8,5))
plt.hist(data["total_sleep"].dropna(), bins=10, color = 'green')

plt.title("Histogram of Total Sleep")
plt.xlabel("Total Sleep (hours)")
plt.ylabel("Frequency")
plt.show()
```

![](Data%20Wrang%20Final_files/figure-commonmark/cell-12-output-1.png)

This code was used to see the distribution of total number of hours of
sleep ranging from 2.5 to 20. With most total hours of sleep falling
into the 7-13 hour range. If a mammal is to not be apart of the 7-13
hour range they are more likely to sleep less hours than more.

## Question 2:

# What features correlate with one another the least?

``` python
corr = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="copper", square=True)
plt.title("Correlation Matrix")
plt.show()
```

![](Data%20Wrang%20Final_files/figure-commonmark/cell-13-output-1.png)

This code was produced to see how all of the features work alongside one
another. The bright red shows the features positively interacting with
one another which is why it is a positive 1.00. The bright blue shows
the negative interaction between the features. The features that
correlate with one another the least is sleep_exposure_index and
total_sleep. Meaning that it matters the least about how exposed the
animal is to the outside elements and the total hours of sleep they get.

## Question 3:

# How does total sleep relate to life span?

``` python
plt.figure(figsize=(8, 6))
plt.scatter(data['total_sleep'], data['max_life_span'], color = 'green')

x = data['total_sleep']
y = data['max_life_span']

m, b = np.polyfit(x, y, 1)

x_sorted = np.sort(x)
y_pred = m * x_sorted + b

plt.plot(x_sorted, y_pred, color='black', linewidth=2)

plt.title("Total Sleep vs. Maximum Life Span")
plt.xlabel("Total Sleep (hours)")
plt.ylabel("Max Life Span (years)")

plt.grid(True, alpha=0.3)
plt.show()
```

![](Data%20Wrang%20Final_files/figure-commonmark/cell-14-output-1.png)

This code was written to compare total sleep vs maximum life span to
take a deeper look into how they correlate. From this scatter plot we
can see that more sleep does not mean that the mammals will have a
longer life span.
