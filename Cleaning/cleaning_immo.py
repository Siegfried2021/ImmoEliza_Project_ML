import pandas as pd
import numpy as np
class DataCleaning:
    """
    A class for cleaning a DataFrame.

    Attributes:
        df (pd.DataFrame): The DataFrame to be cleaned.
    """
    def __init__(self, df):
        """
        Initializes the DataCleaning class with the given DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to be cleaned.
        """
        self.df = df

    def remove_values(self, column, values):
        """
        Sets values in the specified column to None if they are not in the given list.

        Parameters:
            column (str): The name of the column to clean.
            values (list): The list of values to keep.
        """
        self.df[column] = self.df[column].apply(lambda x: x if x in values else None)

    def subset_dataframe(self, column, values):
        """
        Filters the DataFrame to only include rows where the specified column's values are in the given list.

        Parameters:
            column (str): The name of the column to filter on.
            values (list): The list of values to keep.
        """
        self.df = self.df[self.df[column].isin(values)]

    def merge_df(self, right_df, left_col, right_col):
        """
        Merges the DataFrame with another DataFrame on specified columns and removes duplicates based on 'PropertyId'.

        Parameters:
            right_df (pd.DataFrame): The DataFrame to merge with.
            left_col (str): The column in the current DataFrame to join on.
            right_col (str): The column in the right DataFrame to join on.
        """
        self.df = self.df.merge(right_df, how="left", left_on=left_col, right_on=right_col).drop_duplicates(subset="PropertyId")

    def remove_outliers(self, column, lower_bound=None, upper_bound=None, filter_column=None, filter_value=None):
        """
        Removes outliers from the DataFrame based on specified bounds and optional filter conditions.

        Parameters:
            column (str): The column to check for outliers.
            lower_bound (numeric, optional): The lower bound for valid values. Defaults to None.
            upper_bound (numeric, optional): The upper bound for valid values. Defaults to None.
            filter_column (str, optional): The column to apply a filter on. Defaults to None.
            filter_value (any, optional): The value to filter on in the filter_column. Defaults to None.
        """
        if filter_column and filter_value:
            filtered_df = self.df[self.df[filter_column] == filter_value]
            non_null_values = filtered_df[column].notna()
            if lower_bound is not None:
                non_null_values &= (filtered_df[column] >= lower_bound)
            if upper_bound is not None:
                non_null_values &= (filtered_df[column] <= upper_bound)
            outliers = filtered_df[~non_null_values]
            self.df = self.df.drop(outliers.index)
        else:
            non_null_values = self.df[column].notna()
            if lower_bound is not None:
                non_null_values &= self.df[column] >= lower_bound
            if upper_bound is not None:
                non_null_values &= self.df[column] <= upper_bound
            null_values = self.df[column].isna()
            self.df = self.df[non_null_values | null_values]

    def remove_incoherent_values(self, column_to_check, reference_column, threshold, filter_column=None, filter_value=None):
        """
        Removes incoherent values based on a linear relationship between two columns.

        Parameters:
            column_to_check (str): The column to check for incoherent values.
            reference_column (str): The reference column to compare against.
            threshold (float): The threshold for acceptable deviation from the expected values.
            filter_column (str, optional): The column to apply a filter on. Defaults to None.
            filter_value (any, optional): The value to filter on in the filter_column. Defaults to None.
        """
        if filter_column and filter_value:
            filtered_df = self.df[self.df[filter_column] == filter_value]
        else:
            filtered_df = self.df
        non_null_values = filtered_df[[column_to_check, reference_column]].dropna()
        coef = np.polyfit(non_null_values[reference_column], non_null_values[column_to_check], 1)
        expected_values = np.polyval(coef, non_null_values[reference_column])
        deviations = abs(non_null_values[column_to_check] - expected_values)
        outlier_values = deviations > threshold * abs(expected_values)
        self.df = self.df[~self.df.index.isin(filtered_df.index[filtered_df.index.isin(non_null_values[outlier_values].index)])]

    def remove_none_values(self, columns):
        """
        Removes rows where any of the specified columns contain None values.

        Parameters:
            columns (list): The list of columns to check for None values.
        """
        self.df = self.df.dropna(subset=columns)

    def convert_to_numbers(self, column, dict_conversion, filter_column = None, filter_value = None):
        """
        Converts categorical values in the specified column to numerical values using a conversion dictionary.

        Parameters:
            column (str): The name of the column to convert.
            dict_conversion (dict): The dictionary mapping categorical values to numerical values.
            filter_column: The column to apply a filter on. Defaults to None.
            filter_value: The value to filter on in the filter_column. Defaults to None.
        """
        if filter_column and filter_value:
            mask = self.df[filter_column] == filter_value
            self.df.loc[mask, f"{column}_numerical"] = self.df.loc[mask, column].map(dict_conversion)
        else:    
            self.df[f"{column}_numerical"] = self.df[column].map(dict_conversion)

    def drop_column(self, column):
        """
        Drops the specified column from the DataFrame.

        Parameters:
            column (str): The name of the column to drop.
        """
        del self.df[column]

    def replace_none_values(self, column, value):
        """
        Replaces None values in the specified column with a given value.

        Parameters:
            column (str): The name of the column to fill.
            value (any): The value to replace None values with.
        """
        self.df[column] = self.df[column].fillna(value)

    def rename_values(self, column, dict_conversion):
        """
        Renames values in the specified column using a conversion dictionary.

        Parameters:
            column (str): The name of the column to rename values in.
            dict_conversion (dict): The dictionary mapping old values to new values.
        """
        self.df[column] = self.df[column].replace(dict_conversion)

    def modify_other_columns(self, column_to_modify, condition_column, operator, target_value, new_value):
        """
        Modifies values in one column based on a condition applied to another column.

        Parameters:
            column_to_modify (str): The name of the column to modify.
            condition_column (str): The name of the column to apply the condition on.
            operator (str): The operator for the condition ('>', '<', '==').
            target_value (any): The value to compare against in the condition column.
            new_value (any): The new value to set in the column_to_modify if the condition is met.
        """
        condition = (self.df[condition_column] > target_value) if operator == '>' else \
                    (self.df[condition_column] < target_value) if operator == '<' else \
                    (self.df[condition_column] == target_value) if operator == '==' else None
        self.df.loc[condition, column_to_modify] = new_value

    def cleaning_process(self, refnis_conversion, city_revenues, city_density):
        """
        Executes a predefined cleaning process on the DataFrame.

        Parameters:
            refnis_conversion (pd.DataFrame): DataFrame for converting postal codes to Refnis codes.
            city_revenues (pd.DataFrame): DataFrame containing city revenue information.
            city_density (pd.DataFrame): DataFrame containing city population density information.
        """
        columns_to_drop = ["Fireplace", "Url", "Country"]
        for column in columns_to_drop:
            self.drop_column(column)

        columns_to_check_none = ["PostalCode", "Price", "PropertyId", "TypeOfSale", "TypeOfProperty"]
        self.remove_none_values(columns_to_check_none)

        merges = [
            (refnis_conversion, "PostalCode", "Postal code"),
            (city_revenues, "Refnis code", "Refnis"),
            (city_density, "Refnis code", "Refnis")
        ]
        for right_df, left_col, right_col in merges:
            self.merge_df(right_df, left_col, right_col)
            self.drop_column(right_col)

        self.subset_dataframe("TypeOfSale", ['residential_sale', 'residential_monthly_rent'])

        rename_dicts = {
            "FloodingZone": {"RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE": "CIRCUMSCRIBED_FLOOD_ZONE"},
            "PEB": {"A+": "A", "A++": "A", "A_A+": "A"},
            "NumberOfFacades": {1: 2},
            "NumberOfFacades": {0: None},
            "TypeOfProperty": {1: "House", 2: "Apartment"}
        }
        for column, rename_dict in rename_dicts.items():
            self.rename_values(column, rename_dict)
        
        self.remove_values("PEB", ["A", "B", "C", "D", "E", "F", "G"])

        columns_replace_none = ["SwimmingPool", "Furnished", "Garden", "Terrace"]
        for column in columns_replace_none:
            self.replace_none_values(column, 0)

        modify_other_cols_params = [
            ("Garden", "GardenArea", ">", 0, 1),
            ("SurfaceOfPlot", "TypeOfProperty", "==", "Apartment", 0),
            ("NumberOfFacades", "TypeOfProperty", "==", "Apartment", 0),
        ]
        for params in modify_other_cols_params:
            self.modify_other_columns(*params)

        outliers_params = [
            ("Price", 10000, 10000000, "TypeOfSale", "residential_sale"),
            ("Price", 1000, 50000, "TypeOfSale", "residential_monthly_rent"),
            ("ConstructionYear", 1750, 2028),
            ("LivingArea", 5, 5000),
            ("ShowerCount", 0, 10),
            ("ToiletCount", 0, 10),
            ("ShowerCount", 0, 10),
            ("BathroomCount", 0, 10),
            ("NumberOfFacades", 0, 4)
        ]
        for params in outliers_params:
            self.remove_outliers(*params)
               
        incoherent_params = [
            ("LivingArea", "Price", 1, "TypeOfSale", "residential_sale"),
            ("LivingArea", "Price", 1, "TypeOfSale", "residential_monthly_rent"),
            ("LivingArea", "SurfaceOfPlot", 1.5),
            ("BedroomCount", "Price", 1.5),
            ("LivingArea", "BedroomCount", 1.5),
            ("BathroomCount", "BedroomCount", 1.5),
            ("ToiletCount", "BedroomCount", 1.5),
            ("ShowerCount", "BedroomCount", 1.5)
        ]
        for params in incoherent_params:
            self.remove_incoherent_values(*params)

        conversion_dicts = {
            "PEB": {'G': 1, 'F': 2, 'E': 3, 'D': 4, 'C': 5, 'B': 6, 'A': 7},
            "Kitchen": {'NOT_INSTALLED': 1, 'USA_UNINSTALLED': 2, 'INSTALLED': 3, 'USA_INSTALLED': 4, 'SEMI_EQUIPPED': 5, 'USA_SEMI_INSTALLED': 6, 'HYPER_EQUIPPED': 7, 'USA_HYPER_EQUIPPED': 8},
            "FloodingZone": {'RECOGNIZED_FLOOD_ZONE': 1, 'POSSIBLE_FLOOD_ZONE': 2, 'CIRCUMSCRIBED_FLOOD_ZONE': 3, 'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE': 4, 'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE': 5, 'CIRCUMSCRIBED_WATERSIDE_ZONE': 6, 'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE': 7, 'NON_FLOOD_ZONE': 8},
            "StateOfBuilding": {'TO_BE_DONE_UP': 1, 'TO_RESTORE': 2, 'TO_RENOVATE': 3, 'GOOD': 4, 'JUST_RENOVATED': 5, 'AS_NEW': 6},
        }
        for column, conversion_dict in conversion_dicts.items():
            self.convert_to_numbers(column, conversion_dict)
        
        conversion_price_sq = [
        ({"Walloon Brabant": 2465, "Hainaut": 1376, "Namur": 1601, "Liège": 1682, "Luxembourg": 1616, "Brussels": 3241, "Flemish Brabant": 2465, "West Flanders": 2055, "East Flanders": 2189, "Antwerp": 2343, "Limburg": 1865}, "House"),
        ({"Walloon Brabant": 3139, "Hainaut": 1840, "Namur": 2423, "Liège": 2214, "Luxembourg": 2427, "Brussels": 3370, "Flemish Brabant": 3159, "West Flanders": 3803, "East Flanders": 2845, "Antwerp": 2732, "Limburg": 2482}, "Apartment")
        ]    
        for conversion_dict, property_type in conversion_price_sq:
            self.convert_to_numbers("Province", conversion_dict, "TypeOfProperty", property_type)
        
        # Additional interventions to convert plot surfaces of houses equal zo zero to None, as well as to add an upper-bound to plot surface    
        self.df.loc[(self.df['TypeOfProperty'] == 'House') & (self.df['SurfaceOfPlot'] == 0), 'SurfaceOfPlot'] = None
        self.df = self.df[(self.df['SurfaceOfPlot'] <= 10000) | (self.df['SurfaceOfPlot'].isna())]    

data_immoweb = pd.read_json("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/final_dataset.json")
df = pd.DataFrame(data_immoweb)
refnis_conversion = pd.read_excel("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/Conversion Postal code_Refnis code_va01012019.xlsx")
city_revenues = pd.read_excel("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/REVENUS.xlsx")
city_density = pd.read_excel("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/Pop_density_fr.xlsx")

dataclean = DataCleaning(df)
dataclean.cleaning_process(refnis_conversion, city_revenues, city_density)

final_df = dataclean.df

final_csv = final_df.to_csv('/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/final_set.csv', index=False)
final_excel = final_df.to_excel('/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/ImmoElizaML/data/final_set.xlsx', index=False)