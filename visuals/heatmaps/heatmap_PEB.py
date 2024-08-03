# create a heatmap of PEB scores in Belgium

import geopandas
import pandas as pd

geo_df=geopandas.read_file("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/BELGIUM_-_Municipalities.geojson")

geo_df.drop(columns=["OBJECTID","ADMUNAFR","ADMUNADU","ADMUNAGE","arrond"],inplace=True)
geo_df["CODE_INS"]=geo_df["CODE_INS"].astype(int)

refnis_conv=pd.read_excel("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/Conversion Postal code_Refnis code_va01012019.xlsx")

conv_table=refnis_conv.astype(str)[["Refnis code","Postal code"]].melt(id_vars="Refnis code").drop_duplicates().pivot_table(index="Refnis code",values="value", aggfunc=",".join).reset_index().rename(columns={"value":"PostalCodes"})
conv_table["Refnis code"]=conv_table["Refnis code"].astype(int)                                                                     

geo_base = geo_df.merge(conv_table,left_on="CODE_INS",right_on="Refnis code",how="left").drop("CODE_INS",axis=1)

geo_base["Refnis code"].isna().sum()

df=pd.read_csv("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/final_set.csv")
df_sale=df[(df["TypeOfSale"]=="residential_sale")]
df_house_sale =df_sale[(df_sale["TypeOfProperty"] == "House")]

df_ref=pd.merge(df,refnis_conv.rename(columns={"Postal code":"PostalCode"}),on="PostalCode",how="left").drop_duplicates(subset=["PropertyId"],keep="first")

df_PEB=df_ref.groupby(by=["Refnis code","Nom commune"]).PEB_numerical.mean().reset_index().drop_duplicates(subset="Refnis code",keep="first")
df_PEB["PEB_numerical"]=df_PEB.PEB_numerical.apply(lambda x:round(x,2))
df_PEB["Refnis code"]=df_PEB["Refnis code"].astype(int)

geo_price = geo_base.merge(df_PEB,on="Refnis code",how="left")

geo_price.plot(column="PEB_numerical",legend=True,missing_kwds={"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "Missing values",})

geo_price.explore("PEB_numerical",scheme="fisherjenks",legend=True,legend_kws={"caption": "Mean PEB score"})