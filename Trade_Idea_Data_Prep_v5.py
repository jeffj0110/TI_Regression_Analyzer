#
#
# This module will a read data dictionary of Field Names to set up regression analysis from data files.
# The field names are either an Input or Target to a regression model.
#
# The data files (Per Ticker) from Trade Ideas are then read and inserted into this input/target framework.
#
# The regression analysis is done on a per ticker and per target field basis.
# So, the inputs for each targets symbol might be
# different (ie. there is a different model for each target field).
#
# Any completely blank columns are removed to facilitate the data regression analysis.
# Depending on the level of data rows with partially blank entries may or may not be removed.
#
# Version 5 will read all stored input data files from TradeIdeas that are available in a ticker directory
# that have a filename of [ticker][fldname]YYYYMM.csv
#
# Version 5 will generate input and output data files for each target field being modelled
#
import csv
import pprint
import pandas as pd
import math
import glob
import numpy as np
import sys, getopt, os
from os import path
import pathlib
import re, string
from datetime import datetime
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

def Dimension_Reduction(tck, targ_fld, Input_DF, NAN_Threshold_Perc, Target_DF, datamonth) :
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
    if NAN_Threshold_Perc < 1 and NAN_Threshold_Perc > 0 :
       NAN_Threshold = int(NAN_Threshold_Perc*Inp_DF_Length)
       Input_DF = Input_DF.dropna(thresh = NAN_Threshold, axis=1)
       Target_DF = Target_DF.dropna(thresh = NAN_Threshold, axis=1)
    else :
       print("Threshold to filter NAN's is not between 0 and 1")
       return -1, Input_DF, Target_DF

    print("Input and Output DataFrame Dimensions after removing Nan Columns ", tck, " ", Input_DF.shape, Target_DF.shape)
    Inp_DF_Length = len(Input_DF)
    Target_DF_Length = len(Target_DF)

    if Target_DF_Length == 0 or Inp_DF_Length == 0 :
        return -1, Input_DF, Target_DF

    if Inp_DF_Length != Target_DF_Length :
        print("Data Sets Have Different # Of Rows ", tck, " ", targ_fld)
        #Delete rows with missing dates of data
        missing_dates = set(Input_DF['DATE_VALUE']).symmetric_difference(set(Target_DF['DATE_VALUE']))
        for date_val in missing_dates :
            Input_DF = Input_DF.drop(Input_DF[Input_DF['DATE_VALUE'] == date_val].index)
            Target_DF = Target_DF.drop(Target_DF[Target_DF['DATE_VALUE'] == date_val].index)

    # Replace remaining NaN's with zeros.
    Input_DF = Input_DF.fillna(0)
    Target_DF = Target_DF.fillna(0)

    if Target_DF_Length == 0 or Inp_DF_Length == 0 :
        return -1, Input_DF, Target_DF
    else:
        Input_DF.sort_values(by=['DATE_VALUE'], inplace=True)
        Target_DF.sort_values(by=['DATE_VALUE'], inplace=True)

    file_name_str_inp = tck + targ_fld + "_Input_Var"
    file_name_str_targ = tck + targ_fld + "_Output_Var"
    Input_Data_File = re.sub(rf"[{string.punctuation}]", "",
                           file_name_str_inp)  # Need to remove any characters which cause issues with file names
    Output_Data_File = re.sub(rf"[{string.punctuation}]", "",
                           file_name_str_targ)  # Need to remove any characters which cause issues with file names
    Input_Data_File += '.csv'
    Output_Data_File += '.csv'
    Input_DF.to_csv(Input_Data_File, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
    Target_DF.to_csv(Output_Data_File, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')

    return 1, Input_DF, Target_DF

def Print_Results(Regression_Calcs) :

   pprint.pprint(Regression_Calcs)

   return


def PreProcess_Input_Data(tck, inputdir, TI_Dict, datamonth) :

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
    TI_Data_DF = pd.DataFrame()
    if len(datamonth) > 5 :
        file_name_header = tck + "_" + datamonth
        search_string = file_name_header + '.{}'
    else:
        file_name_header = tck + "_202"
        search_string = file_name_header + '???.{}'

    extension = 'csv'
    all_filenames = [i for i in glob.glob(search_string.format(extension))]
    # combine all files in the list that begin with tck_YYYYMM
    if len(all_filenames) > 0:
        print("reading ticker ", tck)
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
        TI_Data_DF = pd.concat([combined_csv, TI_Data_DF], ignore_index=True, sort=False)
    else:
        print("No files for ", tck)

    TI_Data_DF.sort_values('Ticker_Date', inplace=True)

    SymCount = 0
    # Read data file and convert from row based to column based per each set of data by date
    Current_Month = datetime.now().month
    Current_Year = datetime.now().year

    num_rows = len(TI_Data_DF)

    # For each target, create an input and output set of files for regression testing
    # Have to make a pass through the input file for every set of Target / Input Variables
    for targ_fld in TI_Dict.keys() :
        Input_list = TI_Dict[targ_fld]
        if targ_fld.endswith('_Max') :   #The _Max doesn't appear in the input files
            trimmed_targ_fld = targ_fld[:-4]
            Target_Col_Names = ['DATE_VALUE'] + [trimmed_targ_fld]
        else :
            Target_Col_Names = ['DATE_VALUE'] + [targ_fld]
        Input_Column_Names = ['DATE_VALUE'] + Input_list
        Target_Var_DF = pd.DataFrame(columns=Target_Col_Names)
        Input_Var_DF = pd.DataFrame(columns=Input_Column_Names)
        data_row = 0

        for data_row in range(num_rows) :
            #Look for field in Input_Var_DF or Target_Var_DF
            data_field = TI_Data_DF.iloc[data_row]['Alert_Filter_Name']
            data_date = TI_Data_DF.iloc[data_row]['Ticker_Date']
            if data_field in Input_list :
                # add data to input variable  dataframe, create a new row if needed
                # print("Found data field in input", data_field)
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
#                print("Found data field in target", data_field)
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

                            if targ_fld.endswith("_MAX"):
                                if math.isnan(Value_Filter_Max) == False:
                                    new_rec_dictionary[data_field] = Value_Filter_Max
                                else:
                                    new_rec_dictionary[data_field] = math.nan
                            else:
                                if math.isnan(Value_Filter) == False:
                                    new_rec_dictionary[data_field] = Value_Filter
                                else:
                                    new_rec_dictionary[data_field] = math.nan

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
                    #if math.isnan(Value_Filter) == False:
                    #    new_rec_dictionary[data_field] = Value_Alert
                    #else:
                    #    new_rec_dictionary[data_field] = math.nan
                    if targ_fld.endswith("_MAX") :
                        if math.isnan(Value_Filter_Max) == False:
                            new_rec_dictionary[data_field] = Value_Filter_Max
                        else:
                            new_rec_dictionary[data_field] = math.nan
                    else :
                        if math.isnan(Value_Filter) == False:
                            new_rec_dictionary[data_field] = Value_Filter
                        else:
                            new_rec_dictionary[data_field] = math.nan
                    Target_Var_DF = Target_Var_DF.append(new_rec_dictionary, ignore_index=True)

        # Perform dimension reduction on data set to avoid extremely large compute times
        # and to avoid using input variables which have no meaningful impact on the target variables
        return_code, Reduced_Input_DF, Reduced_Target_DF = Dimension_Reduction(tck, targ_fld, Input_Var_DF, .5, \
                                                                               Target_Var_DF, datamonth)

        if return_code < 0 :
            print("Errors with data Pre-Processing ", tck, " ", targ_fld)

    print("Completed Pre-Processing Ticker ", tck)
    SymCount += 1

    return SymCount, Input_Var_DF, Target_Var_DF

# Create a dictionary data structure for Targets and their associated Input fields
def Get_Data_Definitions() :
    # Read in list of possible fields and the defined input/target definition
    TI_Data_Dict = pd.read_csv('Trade-Ideas_Data_Dictionary_Individual_Field_Models.csv', encoding='UTF-8-sig')
    num_flds = len(TI_Data_Dict)
    if num_flds == 0 :
       return None

    # Target = Target Data Fields
    # Each Target has a dictionary of Input Fields
    TI_Dictionary = {}

    for r in range(num_flds) :
        row_data = TI_Data_Dict.iloc[r]
        Target_Fld = row_data['Targets']
        Input_Fld = row_data['Inputs']
        if Target_Fld not in TI_Dictionary:
            TI_Dictionary[Target_Fld] = list()

        if Input_Fld not in TI_Dictionary[Target_Fld] :
            TI_Dictionary[Target_Fld].extend([Input_Fld])

    return TI_Dictionary

# Check to see if this file is being executed as the "Main" python
# script instead of being used as a module by some other python script
# This allows us to use the module which ever way we want.
def main(argv):
   inputfile = 'Ticker_List.csv'
   month_of_data = '202109'
   ID_Counter = 1
   datadir = 'Dnld_Data'
   try:
      opts, args = getopt.getopt(argv,"hi:d:p:",["ifile=","ddir=", "pdate="])
   except getopt.GetoptError:
      print('Trade_Ideas_DP_v5 -i <inputfile> -d <inputdatadir>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Trade_Ideas_DP_v5 -i <inputfile> -d <inputdatadir>')
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

   TI_Dictionary = Get_Data_Definitions()

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

   for tick_row in range(num_tickers):
       # Download indicators for each symbol month to date
       row_tick = Ticker_List.iloc[tick_row]
       tck = row_tick['Ticker']
       os.chdir(CWD)

       return_code, Data_Filled_Input_DF, Data_Filled_Target_DF = PreProcess_Input_Data(tck, datadir, TI_Dictionary, month_of_data)

   return True


if __name__ == "__main__":
   main(sys.argv[1:])