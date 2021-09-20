# This script combines several spreadsheets from data downloads
# from Trade Ideas
#
# The end result is to gain a set of fields that are possibly downloadable
# from Trade Ideas
#
import csv
import pandas as pd
import numpy as np
import sys, getopt, os
from os import path
import pathlib
import re
from datetime import datetime
import time
import glob


def Concatenate_CSV_Files(InputFileName, process_date):
    # Only process files from the current month
    Current_Month = datetime.now().month
    Current_Day = datetime.now().day
    Current_Year = datetime.now().year

    base_directory = pathlib.Path.cwd()
    fname = base_directory.joinpath(InputFileName).resolve()
#    Ticker_List = pd.read_csv(fname, encoding='UTF-8-sig', names=['Ticker'])
    Ticker_List = pd.read_csv(fname, encoding='UTF-8')
    num_tickers = len(Ticker_List)
    CurrentDirectory = os.getcwd()

    # Initialize df to store all CSV data
    Combined_df = pd.DataFrame(columns=['Ticker', 'Ticker_Date', 'Target_Field',
                                        'R2_Value', 'Coefficient Field',
                                        'Coefficient Value'])

    for row in range(num_tickers):
        os.chdir(CurrentDirectory)
        os.chdir("Dnld_Data")
        row_data = Ticker_List.iloc[row]
        tck = row_data['Ticker']
        os.chdir(tck)
        file_name_header = tck + "_InputToTarget_" + process_date + "_regr_results"
        extension = 'csv'
        search_string = file_name_header + '*.{}'
        all_filenames = [i for i in glob.glob(search_string.format(extension))]
        # Do not concatenate to the existing Combined*.csv file, just overwrite it.
        for i in range(len(all_filenames) - 1, -1, -1):
            if '_combined.csv' in all_filenames[i].lower():
                del all_filenames[i]
        # combine all files in the list that begin with tck_YYYYMM

        if len(all_filenames) > 0 :
            print("reading ticker ", tck)
            combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
            Combined_df = pd.concat([combined_csv, Combined_df], ignore_index=True, sort=False)
        else :
            print("No files for ", tck)

    # export to csv
    os.chdir(CurrentDirectory)
    os.chdir("Dnld_Data")
    print("Combined Dataframe Before ", Combined_df.shape)
    Combined_df['Coeff_Abs'] = abs(Combined_df['Coefficient Value'])
    Combined_df = Combined_df[Combined_df.iloc[:, 3].between(0, 1, inclusive=False)]
    Combined_df.sort_values(by=['R2_Value', 'Coeff_Abs'], ascending=[False, False], inplace=True)
    print("Combined Dataframe After ", Combined_df.shape)
    Combined_File_Name = process_date + "_InputToTarget_RegressionResults_Combined.csv"
    Combined_df.to_csv(Combined_File_Name, index=False, encoding='utf-8-sig')
    print("Combined CSV Files Into ", Combined_File_Name)

    return num_tickers


# Check to see if this file is being executed as the "Main" python
# script instead of being used as a module by some other python script
# This allows us to use the module which ever way we want.
def main(argv):
    global ID_Counter
    inputfile = 'Ticker_List.csv'
    ID_Counter = 1
    month_of_data = '202108'
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:p:", ["ifile=","pdate="])
    except getopt.GetoptError:
        print('ConCat_TradeIdea_Files -i <inputfile> -pdate <YYYYMM>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('ConCat_TradeIdea_Files -i <inputfile> -pdate <YYYYMM>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-p", "-pdate"):
            month_of_data = arg

    print('Input file is :', inputfile)
    print("Processing Data From ", month_of_data)
    return_code = Concatenate_CSV_Files(inputfile, month_of_data)

    if return_code > 0:
        print("Number Ticker Files Processed = ", return_code)
    else:
        print("No csv Files Retrieved For ", inputfile)
        return False


if __name__ == "__main__":
    main(sys.argv[1:])