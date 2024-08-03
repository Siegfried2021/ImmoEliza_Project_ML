# create a heatmap of house prices in Belgium

import geopandas
import pandas as pd

# Load the GeoJSON file containing Belgium municipalities
geo_df = gpd.read_file("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/BELGIUM_-_Municipalities.geojson")

# Drop unnecessary columns and convert 'CODE_INS' to integer
geo_df.drop(columns=["OBJECTID", "ADMUNAFR", "ADMUNADU", "ADMUNAGE", "arrond"], inplace=True)
geo_df["CODE_INS"] = geo_df["CODE_INS"].astype(int)

# Load the Excel file for Refnis code to Postal code conversion
refnis_conv = pd.read_excel("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/Conversion Postal code_Refnis code_va01012019.xlsx")

# Transform the conversion table to get a pivot table with Refnis code and corresponding Postal codes
conv_table = refnis_conv.astype(str)[["Refnis code", "Postal code"]].melt(id_vars="Refnis code").drop_duplicates().pivot_table(index="Refnis code", values="value", aggfunc=",".join).reset_index().rename(columns={"value": "PostalCodes"})
conv_table["Refnis code"] = conv_table["Refnis code"].astype(int)

# Merge the geo data with the conversion table to get postal codes for each municipality
geo_base = geo_df.merge(conv_table, left_on="CODE_INS", right_on="Refnis code", how="left").drop("CODE_INS", axis=1)
missing_refnis_count = geo_base["Refnis code"].isna().sum()

# Filter data for residential sales and houses
df = pd.read_csv("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/final_set.csv")
df_sale = df[df["TypeOfSale"] == "residential_sale"]
df_house_sale = df_sale[df_sale["TypeOfProperty"] == "House"]

# Merge the house sale data with the Refnis conversion table and remove duplicates
df_ref = pd.merge(df_house_sale, refnis_conv.rename(columns={"Postal code": "PostalCode"}), on="PostalCode", how="left").drop_duplicates(subset=["PropertyId"], keep="first")

# Calculate mean price per Refnis code and municipality name, round prices to 2 decimal places
df_price = df_ref.groupby(by=["Refnis code", "Nom commune"]).Price.mean().reset_index().drop_duplicates(subset="Refnis code", keep="first")
df_price["Price"] = df_price.Price.apply(lambda x: round(x, 2))
df_price["Refnis code"] = df_price["Refnis code"].astype(int)

# Merge the geo data with the price data
geo_price = geo_base.merge(df_price, on="Refnis code", how="left")

# Plot the data with 'Price' column, highlighting missing values
geo_price.plot(column="Price", legend=True, missing_kwds={"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "Missing values"})

geo_price.explore("Price", scheme="fisherjenks", legend=True, legend_kws={"caption": "Mean prices"})