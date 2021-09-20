#
#
# This module will a read pre-processed data files to set up regression analysis from data files.
# There are input/target data files for each symbol (ie. company).  The data files are pre-processed
# and have non-relevant (ie. NAN's or high % of NAN's) removed.
#
# The regression analysis is done on a per ticker and per target field basis.
# So, the inputs and targets for target field for each symbol might be
# different (ie. there is a different model for each target field being analyzed).
#

import csv
import pprint
import pandas as pd
import math
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
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

def Iterate_Regresssions(tck, targ_fld, Input_DF, Target_DF) :

    Regression_Results = pd.DataFrame(columns=['Ticker_Date', 'Target_Field', 'R2_Value'])
    num_target_columns = len(Target_DF.columns)
    if num_target_columns < 1 :
        return 1, Regression_Results

    ticker_date = int(Target_DF.iloc[0][0])  # First column and first field which is the date
    ticker_month = str(ticker_date)[:6]            # Just take the year and month of the date field
    # The date column is not an input variable
    Input_DF = Input_DF.iloc[: , 1: ]   # this slice selects all rows then all the columns after the first column (deletes column 0)
    len_df = len(Input_DF)
    training_rows = int(len_df * .7)
    test_rows = int(len_df * .3)
    diff_rounding = len_df - (training_rows + test_rows)
    training_rows = training_rows + diff_rounding

    for target_column_counter in range(1,num_target_columns) : # Skipping the date column, which is the 1st column
        Target_Subset = Target_DF.iloc[:, [target_column_counter]] # Regression on each target column
        scaler = preprocessing.StandardScaler()
        scaled_Input_DF = scaler.fit_transform(Input_DF)
        scaled_Target_Subset = scaler.fit_transform(Target_Subset)
        # split the data into training (70%) and testing (30%) sets
        #X_train, X_test, y_train, y_test = train_test_split(scaled_Input_DF, scaled_Target_Subset, test_size=0.3)
        X_train = scaled_Input_DF[0:training_rows, :]
        y_train = scaled_Target_Subset[0:training_rows, :]
        X_test = scaled_Input_DF[training_rows:, :]
        y_test = scaled_Target_Subset[training_rows:, :]

        # train PCR model on training data
        regr = LinearRegression()
        regr.fit(X_train, y_train)

        pred = regr.predict(X_test)

        # calculate R squared using PCA model regression but using the test data actuals
        R2_score = r2_score(y_test, pred)
        if R2_score < 0 :
            R2_value = np.sqrt(R2_score*(-1)) * -1
        else :
            R2_value = np.sqrt(R2_score)
        coeffs = pd.DataFrame({"Feature":Input_DF.columns.tolist(),"Coefficients":regr.coef_[0]})
        field_name = Target_DF.columns[target_column_counter]
        print("R2 Score For Field ", field_name, " ", R2_value)
        num_coeff = len(coeffs)
        for n in range(num_coeff) :
            new_rec = {'Ticker_Date': ticker_month,
               'Target_Field': field_name,
               'R2_Value': R2_value,
               'Coefficient Field' : coeffs.iloc[n,0],
               'Coefficient Value' : coeffs.iloc[n,1]
               }
            Regression_Results = Regression_Results.append(new_rec, ignore_index=True)
        # Untransform actual and predicted values
        Y_untransformed = scaler.inverse_transform(y_test)
        pred_untransformed = scaler.inverse_transform(pred)
        Write_Predictions(Target_DF, tck, field_name, Y_untransformed, pred_untransformed)

    #    print(X_train_pca.explained_variance)
#    print(X_train_pca.explained_variance_ratio_)
    #print("Components for ", tck, " = ", pca.n_components_)
    # Dump components relations with features:
#    pprint.pprint(pd.DataFrame(pca.components_, columns=Input_DF.columns))
    #print("Explained Variance ", pca.explained_variance_ratio_ * 100)

    return target_column_counter, Regression_Results, y_test, pred

# This function will write the regresion results for each target field
# The input parameter is a dataframe of fields and results
def Write_Results(ticker, Regression_Calc_results, direction) :

   print("Writing Regression Results For ", ticker)
   date_string = str(int(Regression_Calc_results.iloc[0][0]))
   file_name_str = ticker + "_" + direction + "_" + date_string + "_regr_results.csv"
#   pprint.pprint(Regression_Calc_results)
   # Each line of the output has a coefficient of the Target field.
   # We sort the list by R2 (high to low), then list the coefficients (high to low)
   # This presents the list with the most significant Targets and their coefficients at the top
   Regression_Calc_results.sort_values(by=['R2_Value', 'Target_Field'], ascending=[False, False], inplace=True)
   Regression_Calc_results.insert(0,'Ticker', ticker)
   # Add absolute value of regression coefficient as the last column

   Regression_Calc_results.to_csv(file_name_str, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')

   return

# This function will write the predictions and the test values for given target field
def Write_Predictions(targ_df, tck, field_name, test_values, predictions) :

   results_len = len(test_values)
   targ_len = len(targ_df)
   start_row = targ_len - results_len
#   print("Writing Predictions Results For ", tck, " ", field_name)
   Actual_versus_Predictions_df = pd.DataFrame()
   Actual_versus_Predictions_df['Date'] = targ_df.iloc[start_row:,0] # All rows of the date column associated with test values
   Actual_versus_Predictions_df['Actual'] =  test_values
   Actual_versus_Predictions_df['Predictions'] = predictions
   uncleaned_file_name_str = tck + field_name + "PREDICTIONS"
   file_name_str = re.sub(rf"[{string.punctuation}]", "", uncleaned_file_name_str) # Need to remove any characters which cause issues with file names
   file_name_str = file_name_str + ".csv"
   Actual_versus_Predictions_df.to_csv(file_name_str, index=False, encoding='utf-8-sig')

   return

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
   ID_Counter = 1
   datadir = 'Dnld_Data'
   try:
      opts, args = getopt.getopt(argv,"hi:d:",["ifile=","ddir="])
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

   print('Input Ticker file is :', inputfile)
   print('Data directory is : ', datadir)

   SymCount = 0
   if os.path.isfile(inputfile):
      base_directory = pathlib.Path.cwd()
      fname = base_directory.joinpath(inputfile).resolve()
      Ticker_List = pd.read_csv(fname, encoding='UTF-8-sig', header=0, names=['Ticker'])
      num_tickers = len(Ticker_List)
   else:
      print("Input File Not Found ", inputfile)
      return SymCount

   TI_Dictionary = Get_Data_Definitions()


   if os.path.isdir(datadir):
       os.chdir(datadir)
   else:
       print("Invalid Input Data Directory  ", inputdir)
       return -1

   CWD = os.getcwd()

   for tick_row in range(num_tickers):
       # process one ticker at a time
       row_tick = Ticker_List.iloc[tick_row]
       tck = row_tick['Ticker']
       os.chdir(CWD)
       os.chdir(tck)

       # For each target, create an input and output set of files for regression testing
       # Have to make a pass through the input file for every set of Target / Input Variables
       Regression_Results_All_Targets = pd.DataFrame(columns=['Ticker_Date', 'Target_Field',
                                        'R2_Value', 'Coefficient Field',
                                        'Coefficient Value'])
       for targ_fld in TI_Dictionary.keys():
           Input_list = TI_Dictionary[targ_fld]

           file_name_str_inp = tck + targ_fld + "_Input_Var"
           file_name_str_targ = tck + targ_fld + "_Output_Var"
           Input_Var_file_name = re.sub(rf"[{string.punctuation}]", "",
                                    file_name_str_inp)  # Need to remove any characters which cause issues with file names
           Target_Var_file_name = re.sub(rf"[{string.punctuation}]", "",
                                     file_name_str_targ)  # Need to remove any characters which cause issues with file names
           Input_Var_file_name += '.csv'
           Target_Var_file_name += '.csv'


           if os.path.isfile(Input_Var_file_name) and os.path.isfile(Target_Var_file_name) :
               Input_DF = pd.read_csv(Input_Var_file_name, encoding='UTF-8-sig')
               Target_DF = pd.read_csv(Target_Var_file_name, encoding='UTF-8-sig')
           else:
               print("No Input/Target Files Found For Ticker : ", tck, ", Field ", targ_fld)
               continue

           # Perform regression on every target field individually
           returncode, Regression_Results, test_values, predictions = Iterate_Regresssions(tck, targ_fld, Input_DF, Target_DF)
           if returncode < 0 :
              print("Errors with Regressions ", tck)
              return False
           Regression_Results_All_Targets = pd.concat([Regression_Results_All_Targets, Regression_Results], ignore_index=True, sort=False)

       Write_Results(tck, Regression_Results_All_Targets, "InputToTarget")


   return True


if __name__ == "__main__":
   main(sys.argv[1:])