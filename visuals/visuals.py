import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

final_df = pd.read_csv("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/final_set.csv")

# 2-by-2 correlations for numerical values (with visual matrix)
numerical_df = final_df.select_dtypes(include=['number', 'bool'])
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",annot_kws={"size": 10}, cbar_kws={"shrink": 0.75})

# correlation between construction year and PEB
PEB_encoding = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
PEB_values = ["A", "B", "C", "D", "E", "F", "G"]
final_df["PEB_numerical"] = final_df["PEB"].map(PEB_encoding)
filtered_df = final_df[final_df['ConstructionYear'] >= 1800]
correlation_PEB_Year = filtered_df[["PEB_numerical", "ConstructionYear"]].corr()

# boxplot displaying relationship between year and PEB
sns.set_style(style="whitegrid")
palette = "muted"
plt.figure(figsize=(12, 8))
sns.boxplot(data=filtered_df, x="PEB", y="ConstructionYear", order=PEB_values, hue="PEB", palette=palette, dodge=False)
plt.xlabel("PEB Score", fontsize=15, fontweight='bold')
plt.ylabel("Construction Year", fontsize=15, fontweight='bold')
plt.title("Relationship Between Properties Construction Year and Energy Performance", fontsize=18, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xticks(rotation=45, fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.gca().set_facecolor('#f7f7f7')
plt.tight_layout()

# relationship between house price for sale and presence/absence of a swimming pool
final_df_residential = final_df[final_df['TypeOfSale'] == 'residential_sale']
final_df_house_for_sale = final_df_residential[final_df_residential['TypeOfProperty'] == "House"]
corr_coefficient_pool_price, p_value_pool_price = stats.pointbiserialr(final_df_house_for_sale["SwimmingPool"], final_df_house_for_sale["Price"])
plt.figure(figsize=(10, 6))
sns.boxplot(data=final_df_house_for_sale, x="SwimmingPool", y="Price")
plt.xlabel("Swimming Pool")
plt.ylabel("House Price")

# proportion of swimming pool per house for sale across provinces
proportion_pool_province = final_df_house_for_sale.groupby('Province')['SwimmingPool'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
proportion_pool_province.plot(kind='bar', color='skyblue')
plt.xlabel("Province")
plt.ylabel("Proportion of houses with swimming pool")
plt.title("Proportion of Houses with Swimming Pool per Province")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# house price based on number of facades
plt.figure(figsize=(10, 6))
sns.boxplot(data=final_df, x="NumberOfFacades", y="Price")
plt.xlabel("Number of facades")
plt.ylabel("House Price")

# house and appartment price per province
price_per_province = final_df.groupby(["Province", "TypeOfProperty"])["Price"].mean().astype(float).reset_index()
pivot_table = price_per_province.pivot(index='Province', columns='TypeOfProperty', values='Price')
pivot_table.plot(kind='bar', color=['purple', 'steelblue'])
plt.xlabel("Province")
plt.ylabel("Mean Price of Properties")
plt.title("Mean price of properties per province by property type")
plt.xticks(rotation=45)

# house price based on plot area versus living area
final_df_house = final_df[final_df["TypeOfProperty"] == "House"]
final_df_house_for_sale = final_df_house[final_df_house["TypeOfSale"] == "residential_sale"]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

sns.scatterplot(x='LivingArea', y='Price', data=final_df_house_for_sale, color='purple', ax=axes[0])
sns.regplot(x='LivingArea', y='Price', data=final_df_house_for_sale, scatter=False, color='purple', ax=axes[0])
axes[0].set_title('Living Area vs Price')
axes[0].set_xlabel('Living Area')
axes[0].set_ylabel('Price')

sns.scatterplot(x='SurfaceOfPlot', y='Price', data=final_df_house_for_sale, color='steelblue', ax=axes[1])
sns.regplot(x='SurfaceOfPlot', y='Price', data=final_df_house_for_sale, scatter=False, color='steelblue', ax=axes[1])
axes[1].set_title('Surface of Plot vs Price')
axes[1].set_xlabel('Surface of Plot')
axes[1].set_ylabel('Price')

axes[1].set_xscale('log')

fig.suptitle('Comparison of Surface and Living Areas as Predictors on Price with Regression Lines', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# bar chart with proportion of properties classified by number of facades per district
facade_counts = final_df_house_for_sale.groupby(['District', 'NumberOfFacades']).size().unstack(fill_value=0)
facade_proportions = facade_counts.div(facade_counts.sum(axis=1), axis=0)
facade_proportions = facade_proportions.sort_values(by=4, ascending=False)

ax = facade_proportions.plot(kind='bar', stacked=True, figsize=(10, 7), colormap='viridis')
ax.set_xlabel('Province')
ax.set_ylabel('Proportion of Facades')
ax.set_title('Proportion of Number of Facades in Each Belgian Province')

plt.legend(title='Number of Facades', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# scatter plot of price based on surface of the plot/living area, discriminating between Wallonia and Flanders
final_df_fw_house_for_sale = final_df_house_for_sale[final_df_house_for_sale['Region'].isin(['Flanders', 'Wallonie'])]

sns.set_style(style="whitegrid")
palette = {'Flanders': 'red', 'Wallonie': 'green'}
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

sns.scatterplot(ax=axes[0], x='SurfaceOfPlot', y='Price', data=final_df_fw_house_for_sale, hue='Region', palette=palette)
sns.regplot(ax=axes[0], x='SurfaceOfPlot', y='Price', data=final_df_fw_house_for_sale[final_df_fw_house_for_sale['Region'] == 'Flanders'], scatter=False, color='blue', line_kws={"label":"Flanders"})
sns.regplot(ax=axes[0], x='SurfaceOfPlot', y='Price', data=final_df_fw_house_for_sale[final_df_fw_house_for_sale['Region'] == 'Wallonie'], scatter=False, color='green', line_kws={"label":"Wallonie"})
axes[0].set_title('House Price vs. Plot Surface', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Plot Surface (m²)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price (€)', fontsize=14, fontweight='bold')
axes[0].legend()

sns.scatterplot(ax=axes[1], x='LivingArea', y='Price', data=final_df_fw_house_for_sale, hue='Region', palette=palette)
sns.regplot(ax=axes[1], x='LivingArea', y='Price', data=final_df_fw_house_for_sale[final_df_fw_house_for_sale['Region'] == 'Flanders'], scatter=False, color='blue', line_kws={"label":"Flanders"})
sns.regplot(ax=axes[1], x='LivingArea', y='Price', data=final_df_fw_house_for_sale[final_df_fw_house_for_sale['Region'] == 'Wallonie'], scatter=False, color='green', line_kws={"label":"Wallonie"})
axes[1].set_title('House Price vs. Living Area', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Living Area (m²)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Price (€)', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()

# display graphs
plt.show()