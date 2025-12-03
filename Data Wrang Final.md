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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">body_weight</th>
<th data-quarto-table-cell-role="th">brain_weight</th>
<th data-quarto-table-cell-role="th">max_life_span</th>
<th data-quarto-table-cell-role="th">gestation_time</th>
<th data-quarto-table-cell-role="th">predation_index</th>
<th data-quarto-table-cell-role="th">sleep_exposure_index</th>
<th data-quarto-table-cell-role="th">danger_index</th>
<th data-quarto-table-cell-role="th">total_sleep</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">57</td>
<td>2.000</td>
<td>12.3</td>
<td>7.5</td>
<td>200</td>
<td>3</td>
<td>1</td>
<td>3</td>
<td>5.4</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">58</td>
<td>0.104</td>
<td>2.5</td>
<td>2.3</td>
<td>46</td>
<td>3</td>
<td>2</td>
<td>2</td>
<td>15.8</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">59</td>
<td>4.190</td>
<td>58.0</td>
<td>24</td>
<td>210</td>
<td>4</td>
<td>3</td>
<td>4</td>
<td>10.3</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">60</td>
<td>3.500</td>
<td>3.9</td>
<td>3</td>
<td>14</td>
<td>2</td>
<td>1</td>
<td>1</td>
<td>19.4</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">61</td>
<td>4.050</td>
<td>17.0</td>
<td>13</td>
<td>38</td>
<td>3</td>
<td>1</td>
<td>1</td>
<td>?</td>
</tr>
</tbody>
</table>

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">body_weight</th>
<th data-quarto-table-cell-role="th">brain_weight</th>
<th data-quarto-table-cell-role="th">max_life_span</th>
<th data-quarto-table-cell-role="th">gestation_time</th>
<th data-quarto-table-cell-role="th">predation_index</th>
<th data-quarto-table-cell-role="th">sleep_exposure_index</th>
<th data-quarto-table-cell-role="th">danger_index</th>
<th data-quarto-table-cell-role="th">total_sleep</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">count</td>
<td>62.000000</td>
<td>62.000000</td>
<td>62</td>
<td>62</td>
<td>62.000000</td>
<td>62.000000</td>
<td>62.000000</td>
<td>62</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">unique</td>
<td>NaN</td>
<td>NaN</td>
<td>48</td>
<td>50</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>45</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">top</td>
<td>NaN</td>
<td>NaN</td>
<td>?</td>
<td>?</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>?</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">freq</td>
<td>NaN</td>
<td>NaN</td>
<td>4</td>
<td>4</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>4</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">mean</td>
<td>198.789984</td>
<td>283.134194</td>
<td>NaN</td>
<td>NaN</td>
<td>2.870968</td>
<td>2.419355</td>
<td>2.612903</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">std</td>
<td>899.158011</td>
<td>930.278942</td>
<td>NaN</td>
<td>NaN</td>
<td>1.476414</td>
<td>1.604792</td>
<td>1.441252</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">min</td>
<td>0.005000</td>
<td>0.140000</td>
<td>NaN</td>
<td>NaN</td>
<td>1.000000</td>
<td>1.000000</td>
<td>1.000000</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">25%</td>
<td>0.600000</td>
<td>4.250000</td>
<td>NaN</td>
<td>NaN</td>
<td>2.000000</td>
<td>1.000000</td>
<td>1.000000</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">50%</td>
<td>3.342500</td>
<td>17.250000</td>
<td>NaN</td>
<td>NaN</td>
<td>3.000000</td>
<td>2.000000</td>
<td>2.000000</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">75%</td>
<td>48.202500</td>
<td>166.000000</td>
<td>NaN</td>
<td>NaN</td>
<td>4.000000</td>
<td>4.000000</td>
<td>4.000000</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">max</td>
<td>6654.000000</td>
<td>5712.000000</td>
<td>NaN</td>
<td>NaN</td>
<td>5.000000</td>
<td>5.000000</td>
<td>5.000000</td>
<td>NaN</td>
</tr>
</tbody>
</table>

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

## Question 1

# How spread out is the total hours of sleep for mammals?

``` python
plt.figure(figsize=(8,5))
plt.hist(data["total_sleep"].dropna(), bins=10, color = 'green')

plt.title("Histogram of Total Sleep")
plt.xlabel("Total Sleep (hours)")
plt.ylabel("Frequency")
plt.show()
```

![](Data%20Wrang%20Final_files/figure-markdown_strict/cell-12-output-1.png)

This code was used to see the distribution of total number of hours of
sleep ranging from 2.5 to 20. With most total hours of sleep falling
into the 7-13 hour range. If a mammal is to not be apart of the 7-13
hour range they are more likely to sleep less hours than more.

## Question 2

# What features correlate with one another the least?

``` python
corr = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="copper", square=True)
plt.title("Correlation Matrix")
plt.show()
```

![](Data%20Wrang%20Final_files/figure-markdown_strict/cell-13-output-1.png)

This code was produced to see how all of the features work alongside one
another. The bright red shows the features positively interacting with
one another which is why it is a positive 1.00. The bright blue shows
the negative interaction between the features. The features that
correlate with one another the least is sleep_exposure_index and
total_sleep. Meaning that it matters the least about how exposed the
animal is to the outside elements and the total hours of sleep they get.

## Question 3

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

![](Data%20Wrang%20Final_files/figure-markdown_strict/cell-14-output-1.png)

This code was written to compare total sleep vs maximum life span to
take a deeper look into how they correlate. From this scatter plot we
can see that more sleep does not mean that the mammals will have a
longer life span.
