#
#
# This module will a read data dictionary of Field Names to set up regression analysis from data files.
# The field names are either an Input or Target to a regression model.
#
# The data files (Per Ticker) from Trade Ideas are then read and inserted into this input/target framework.
#
# The regression analysis is done on a per ticker basis.  So, the inputs and targets for each ticker symbol might be
# different (ie. there is a different model for each company being analyzed).
#
# Any completely blank columns are removed to facilitate the data regression analysis.
# Depending on the level of data rows with partially blank entries may or may not be removed.
#
import csv
import pprint
import pandas as pd
import math
import json
import numpy as np
import sys, getopt, os
from os import path
import pathlib
import re
from datetime import datetime
import time


def Dimension_Reduction(tck, Input_DF, NAN_Threshold_Perc, Target_DF, datamonth) :
    # Look for NAN's in the input dataset.
    # We need to either assign them a value or delete the row or column.
    #
    # Drops all columns which are all NAN's

    print("Input and Output DataFrame Dimensions Before Reduction for ", tck, " ", Input_DF.shape, Target_DF.shape)

    Input_DF = Input_DF.dropna(axis=1, how='all')
    Target_DF = Target_DF.dropna(axis=1, how='all')

    # Drops all columns which have more than NAN_Threshold_Perc NAN's
    # The parameter NAN_Threshold_Perc needs to be a decimal percentage (ie. .7 = 70%)
    Inp_DF_Length = len(Input_DF)
    Target_DF_Length = len(Target_DF)
    if NAN_Threshold_Perc < 1 and NAN_Threshold_Perc > 0 :
       NAN_Threshold = int(NAN_Threshold_Perc*Inp_DF_Length)
       Input_DF = Input_DF.dropna(thresh = NAN_Threshold, axis=1)
       Target_DF = Target_DF.dropna(thresh = NAN_Threshold, axis=1)
    else :
       print("Threshold to filter NAN's is not between 0 and 1")
       return -1, Input_DF, Target_DF

    print("Input and Output DataFrame Dimensions after removing Nan Columns ", tck, " ", Input_DF.shape, Target_DF.shape)

    # Replace remaining NaN's with zeros.
    Input_DF = Input_DF.fillna(0)
    Target_DF = Target_DF.fillna(0)

    Target_DF_Length = len(Target_DF)
    Inp_DF_Length = len(Input_DF)
    if Target_DF_Length == 0 or Inp_DF_Length == 0 :
        return -1, Input_DF, Target_DF
    else:
        Input_DF.sort_values(by=['DATE_VALUE'], inplace=True)
        Target_DF.sort_values(by=['DATE_VALUE'], inplace=True)

    Input_DF.to_csv(tck + "_Input_Var.csv", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
    Target_DF.to_csv(tck + "_Output_Var.csv", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')

    return 1, Input_DF, Target_DF

def Print_Results(Regression_Calcs) :

   pprint.pprint(Regression_Calcs)

   return


def PreProcess_Input_Data(tck, inputdir, Input_Var_DF, Target_Var_DF, datamonth) :

    if os.path.isdir(inputdir) :
        os.chdir(inputdir)
    else :
        print("Invalid Input Data Directory  ", inputdir)
        return -1

    if os.path.isdir(tck) :
        os.chdir(tck)
    else :
        print("Invalid Ticker - No Data Directory ")
        return -1

    SymCount = 0
    # Read data file and convert from row based to column based per each set of data by date
    Current_Month = datetime.now().month
    Current_Year = datetime.now().year
    data_file_name = tck + "_" + datamonth + ".csv"
    TI_Data_DF = pd.read_csv(data_file_name, encoding='UTF-8-sig')
    num_rows = len(TI_Data_DF)
    for data_row in range(num_rows) :
        #Look for field in Input_Var_DF or Target_Var_DF
        data_field = TI_Data_DF.iloc[data_row]['Alert_Filter_Name']
        data_date = TI_Data_DF.iloc[data_row]['Ticker_Date']
        if data_field in Input_Var_DF.columns :
            # add data to input variable  dataframe, create a new row if needed
#            print("Found data field in input", data_field)
            # Extract date from field
            date_YYYYMMDD = data_date[-8:]
            boolean_findings = Input_Var_DF['DATE_VALUE'].str.contains(date_YYYYMMDD)
            total_occurence = boolean_findings.sum()
            if total_occurence > 0 :
                cntr = len(boolean_findings)
                row_counter = 0
                while row_counter < cntr :
                    if boolean_findings[row_counter] :
                        Value_Filter = float(TI_Data_DF.iloc[data_row]['Filter_Value1'])
                        Value_Alert = float(TI_Data_DF.iloc[data_row]['Alert_Value'])
                        #Value_Alert = Value_Filter
                        if math.isnan(Value_Alert) == False:
                            Input_Var_DF.at[row_counter, data_field] = Value_Alert
                        else :
                            Input_Var_DF.at[row_counter, data_field] = math.nan
                    row_counter += 1
            else :   # Add new record to dataframe
                new_rec_fld_list = Input_Var_DF.columns.values.tolist()
                new_rec_dictionary = {}
                for i in range(len(new_rec_fld_list)):
                    new_rec_dictionary[new_rec_fld_list[i]] = None
                new_rec_dictionary['DATE_VALUE'] = date_YYYYMMDD
                Value_Filter = float(TI_Data_DF.iloc[data_row]['Filter_Value1'])
                Value_Alert = float(TI_Data_DF.iloc[data_row]['Alert_Value'])
                #Value_Alert = Value_Filter
                if math.isnan(Value_Alert) == False:
                    new_rec_dictionary[data_field] = Value_Alert
                else:
                    new_rec_dictionary[data_field] = math.nan
                Input_Var_DF = Input_Var_DF.append(new_rec_dictionary, ignore_index=True)
        elif data_field in Target_Var_DF.columns :
            # add data to target variable df, create a new row if needed
#            print("Found data field in target", data_field)
            date_YYYYMMDD = data_date[-8:]
            boolean_findings = Target_Var_DF['DATE_VALUE'].str.contains(date_YYYYMMDD)
            total_occurence = boolean_findings.sum()
            if total_occurence > 0:
                cntr = len(boolean_findings)
                row_counter = 0
                while row_counter < cntr:
                    if boolean_findings[row_counter]:
                        Value_Alert = float(TI_Data_DF.iloc[data_row]['Alert_Value'])
                        Value_Filter = float(TI_Data_DF.iloc[data_row]['Filter_Value1'])
                        Value_Filter_Max = float(TI_Data_DF.iloc[data_row]['Filter_Value2'])
                        if math.isnan(Value_Filter) == False :
                            Target_Var_DF.at[row_counter, data_field] = Value_Filter
                        else:
                            Target_Var_DF.at[row_counter, data_field] = math.nan
                        if math.isnan(Value_Filter_Max) == False:
                            Target_Var_DF.at[row_counter, str(data_field+'_Max')] = Value_Filter_Max
                        else:
                            Target_Var_DF.at[row_counter, str(data_field+'_Max')] = math.nan
                    row_counter += 1
            else:  # Add new record to dataframe
                new_rec_fld_list = Target_Var_DF.columns.values.tolist()
                new_rec_dictionary = {}
                for i in range(len(new_rec_fld_list)):
                    new_rec_dictionary[new_rec_fld_list[i]] = None
                new_rec_dictionary['DATE_VALUE'] = date_YYYYMMDD
                Value_Alert = float(TI_Data_DF.iloc[data_row]['Alert_Value'])
                Value_Filter = float(TI_Data_DF.iloc[data_row]['Filter_Value1'])
                Value_Filter_Max = float(TI_Data_DF.iloc[data_row]['Filter_Value2'])
                if math.isnan(Value_Filter) == False:
                    new_rec_dictionary[data_field] = Value_Alert
                else:
                    new_rec_dictionary[data_field] = math.nan
                if math.isnan(Value_Filter_Max) == False:
                    new_rec_dictionary[data_field] = Value_Filter_Max
                else:
                    new_rec_dictionary[data_field] = math.nan
                Target_Var_DF = Target_Var_DF.append(new_rec_dictionary, ignore_index=True)
#        else :
#            print("Unknown data field ", data_field, " for ticker ", tck)


    print("Processed Ticker ", tck)
    SymCount += 1

    return SymCount, Input_Var_DF, Target_Var_DF

def Get_Data_Definitions() :
    # Read in list of possible fields and the defined input/target definition
    TI_Data_Dict = pd.read_csv('Trade-Ideas_Data_Dictionary.csv', encoding='UTF-8-sig')
    num_flds = len(TI_Data_Dict)
    if num_flds == 0 :
       return None, None
    Input_Column_Names = ['DATE_VALUE']
    Target_Column_Names = ['DATE_VALUE']
    for r in range(num_flds) :
        row_data = TI_Data_Dict.iloc[r]
        input_target = row_data['Input_Target']
        Field_Name = row_data['Alert_Filter_Name']

        if input_target == 'Input' :
           Input_Column_Names += [Field_Name]
        elif input_target == 'Target' :
           Target_Column_Names += [Field_Name]
           Target_Column_Names += [Field_Name+"_Max"]
        else:
           print("Invalid Data Dictionary Entry ", Field_Name)

    Input_DF = pd.DataFrame(columns=Input_Column_Names)
    Target_DF = pd.DataFrame(columns=Target_Column_Names)
    return Input_DF, Target_DF

# Check to see if this file is being executed as the "Main" python
# script instead of being used as a module by some other python script
# This allows us to use the module which ever way we want.
def main(argv):
   inputfile = 'Ticker_List.csv'
   month_of_data = '202108'
   ID_Counter = 1
   datadir = 'Dnld_Data'
   try:
      opts, args = getopt.getopt(argv,"hi:d:p:",["ifile=","ddir=", "pdate="])
   except getopt.GetoptError:
      print('Trade_Ideas_Data_Prep -i <inputfile> -d <inputdatadir>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Trade_Ideas_Data_Prep -i <inputfile> -d <inputdatadir>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-d", "--ddir"):
         datadir = arg
      elif opt in ("-p", "--pdate"):
         month_of_data = arg

   print('Input file is :', inputfile)
   print('Data directory is : ', datadir)
   print('Processing Data From ', month_of_data)

   Input_DF, Target_DF = Get_Data_Definitions()

   SymCount = 0
   if os.path.isfile(inputfile):
      base_directory = pathlib.Path.cwd()
      fname = base_directory.joinpath(inputfile).resolve()
      Ticker_List = pd.read_csv(fname, encoding='UTF-8-sig', header=0, names=['Ticker'])
      num_tickers = len(Ticker_List)
   else:
      print("Input File Not Found ", inputfile)
      return SymCount

   CWD = os.getcwd()
   Common_Features_Input = {}
   Common_Features_Target = {}

   for tick_row in range(num_tickers):
       # Download indicators for each symbol month to date
       row_tick = Ticker_List.iloc[tick_row]
       tck = row_tick['Ticker']
       os.chdir(CWD)

       return_code, Data_Filled_Input_DF, Data_Filled_Target_DF = PreProcess_Input_Data(tck, datadir, Input_DF, Target_DF, month_of_data)

       # Perform dimension reduction on data set to avoid extremely large compute times
       # and to avoid using input variables which have no meaningful impact on the target variables
       if return_code > 0 :
          return_code, Reduced_Input_DF, Reduced_Target_DF = Dimension_Reduction(tck, Data_Filled_Input_DF, .7, Data_Filled_Target_DF, month_of_data)
       else :
          print("Errors with data Pre-Processing", return_code)
          return False

       if (len(Common_Features_Input) == 0) :
           Common_Features_Input = set(Reduced_Input_DF.columns)
           Common_Features_Target = set(Reduced_Target_DF.columns)
       else:
           Common_Features_Input = Common_Features_Input.intersection(set(Reduced_Input_DF.columns))
           Common_Features_Target = Common_Features_Target.intersection(set(Reduced_Target_DF.columns))

   # Write out common input and target features
   os.chdir(CWD)
   print ("Length of input features = ", len(Common_Features_Input))
   print ("Length of target list = ", len(Common_Features_Target))
   with open('common_input_features.txt', 'w') as in_feat :
       in_feat.write(json.dumps(list(Common_Features_Input)))

   with open('common_target_features.txt', 'w') as targ_feat :
       targ_feat.write(json.dumps(list(Common_Features_Target)))

   return True


if __name__ == "__main__":
   main(sys.argv[1:])