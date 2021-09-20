# Visualize prediction data
#
#
# This uses 'plotly' to create graphs from the regression predictions.  There is one PDF file
# generated for the regression results for each ticker.
# The PDF file is written to the directory that the regression results are in.
#
import csv
import pandas as pd
import numpy as np
import sys, getopt, os
from os import path
import pathlib
import re, string
from datetime import datetime
import time
import glob

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Create a set of subplots for the ticker and write to a pdf file
def plot_write_predictions(tick, TI_Dict, R2s, Predictions ) :

    plot_file_name = tick + "_plot_predictions.pdf"
    with PdfPages(plot_file_name) as pdf :
        n=0
        fig = plt.figure(figsize=(12,12))
        for targ_fld in TI_Dict.keys():
            # Extract R2 For Field
            if 'Target_Field' in R2s.columns :
                R2s = R2s.drop_duplicates(subset='Target_Field', keep='first')
            else :
                R2s = pd.DataFrame() # Invalid set of R2s
            if len(R2s) > 0 :
                if targ_fld.endswith('_Max'):  # The _Max doesn't appear in the input files
                    trimmed_targ_fld = targ_fld[:-4]
                    R2_Value_Row = R2s.loc[R2s['Target_Field'] == trimmed_targ_fld ]
                else:
                    R2_Value_Row = R2s.loc[R2s['Target_Field'] == targ_fld]
                R2_Value = float(R2_Value_Row['R2_Value'])
            else :
                R2_Value = 0.0

            if ('Field' in Predictions.columns) :
                Field_Prediction_Rows =  Predictions.loc[Predictions['Field'] == targ_fld]
                if len(Field_Prediction_Rows) > 0:
                    date_list = Field_Prediction_Rows['Date']
                    dates_xaxis = [datetime.date(2021,9,xx+3) for xx in range(len(date_list))]
                    date_cnt = 0
                    for dateval in date_list :
                        date_string = str(dateval)
                        date_obj = datetime.date(int(date_string[0:4]), int(date_string[4:6]), int(date_string[6:8]))
                        dates_xaxis[date_cnt] = date_obj
                        date_cnt += 1

                    Actual_vals = Field_Prediction_Rows['Actual']
                    Pred_vals = Field_Prediction_Rows['Predictions']
                    n += 1
                    ax = fig.add_subplot(6,2,n)
                    myFmt = mdates.DateFormatter('%m/%d/%Y')
                    ax.xaxis.set_major_formatter(myFmt)
                    ax.xaxis.set_major_locator(mdates.DayLocator())
                    ax.plot(dates_xaxis, Actual_vals, linewidth=3, label='Actual')
                    ax.plot(dates_xaxis, Pred_vals, linewidth=3, label='Prediction')
                    ax.title.set_text(targ_fld + ', R2=' + str(round(R2_Value,2)))
                    ax.legend()
                    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#                    plt.setp(ax.xaxis.get_ticklabels(), rotation='45')

        fig.subplots_adjust(hspace=1.25)
        fig.suptitle("Actuals / Predictions For " + tick)
        pdf.savefig(fig)
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
      print('Trade_Ideas_Visualize_Results -i <inputfile> -d <inputdatadir>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Trade_Ideas_Visualize_Results -i <inputfile> -d <inputdatadir>')
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
       print("Graphing ", tck)
       # For each Ticker, read the R2 regression results
       file_name_str = tck + "_InputToTarget" + "_" + "202109" + "_regr_results.csv"
       R2_DF = pd.read_csv(file_name_str, encoding='UTF-8-sig')
       if os.path.isfile(file_name_str) :
           R2_DF = pd.read_csv(file_name_str, encoding='UTF-8-sig')
       else:
           print("No Regression Files Found For Ticker : ", tck)
           continue

       # For each target, create read the predictions and concatenate into a df
       All_Predictions = pd.DataFrame()
       for targ_fld in TI_Dictionary.keys():
           # Read in all Predictions
           if targ_fld.endswith('_Max'):  # The _Max doesn't appear in the input files
               trimmed_targ_fld = targ_fld[:-4]
               uncleaned_file_name_str = tck + trimmed_targ_fld + "PREDICTIONS"
           else:
               uncleaned_file_name_str = tck + targ_fld + "PREDICTIONS"
           file_name_str = re.sub(rf"[{string.punctuation}]", "",
                                  uncleaned_file_name_str)  # Need to remove any characters which cause issues with file names
           file_name_str = file_name_str + ".csv"
           if os.path.isfile(file_name_str):
               Pred_DF = pd.read_csv(file_name_str, encoding='utf-8-sig')
               # Concatenate all the predictions
               Pred_DF.insert(0, 'Ticker', tck)
               Pred_DF.insert(1, 'Field', targ_fld)
               All_Predictions = pd.concat([Pred_DF, All_Predictions], ignore_index=True, sort=False)
           else:
               print("No Prediction File Found For Ticker : ", tck, " Target Field ", targ_fld)
               continue


       # For each target, create create a plot
       if len(All_Predictions) > 0 :
           plot_write_predictions(tck, TI_Dictionary, R2_DF, All_Predictions )
       else :
           print("No Predictions Graphed For ", tck)


   return True


if __name__ == "__main__":
   main(sys.argv[1:])