import csv
import pandas as pd
import numpy as np
import sys, getopt, os
from os import path
import pathlib
import re
from datetime import datetime
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup

#base URL which will have symbols inserted to make repeated requests
base_url = "https://www.trade-ideas.com/StockInfo/_StocksLikeThis.html?symbol=REPLACE_SYMBOL&start_date=REPLACE_SECONDS&start_hours=10&start_minutes=30&time=60"

def Valid_Alert(Alert_Name_List, Alert) :
    List_Of_Alerts = Alert_Name_List['Alert']
    for alert_item in List_Of_Alerts :
        if alert_item == Alert :
            return True
    return False

def Get_Data(Alert_Name_List, inputfile, Dnld_Directory) :
    SymCount = 0



    if os.path.isfile(inputfile):
        base_directory = pathlib.Path.cwd()
        fname = base_directory.joinpath(inputfile).resolve()
        Ticker_List = pd.read_csv(fname)
        num_tickers = len(Ticker_List)
    else:
        print("Input File Not Found ", inputfile )
        return SymCount

    os.makedirs(Dnld_Directory, exist_ok=True)
    os.chdir(Dnld_Directory)
    CWD = os.getcwd()

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers['User-Agent'] = 'Financial Inference Technology jeffjones@fitsolutionsusa.com'

    for row in range(num_tickers):
        # Download indicators for each symbol month to date
        row_tick = Ticker_List.iloc[row]
        tck = row_tick['Ticker']
        os.chdir(CWD)
        os.makedirs(tck, exist_ok=True)
        os.chdir(tck)
        # Cycle through every trading day of this month
        Current_Month = datetime.now().month
        Current_Day = datetime.now().day
        Current_Year = datetime.now().year
        day_cnt = Current_Day

        # Initialize df to store values
        alert_values = pd.DataFrame(columns=['Ticker_Date', 'Alert_Filter_Name', 'Alert_Value',
                                             'Filter_Value1', 'Filter_Value2'])
        while day_cnt > 0 :
            # Calc UNIX Epoch time for HTTP call to Trade Ideas
            dt = datetime(Current_Year, Current_Month, day_cnt,3,0)
            Epoch_time = int(dt.timestamp())
            # insert symbol and date into request url
            new_url_tck = base_url.replace('REPLACE_SYMBOL',tck, 1)
            url_to_sub = new_url_tck.replace('REPLACE_SECONDS', str(Epoch_time),1)
            time.sleep(1)    # Delay 1 second to avoid over running website
            response = session.get(url_to_sub)
            return_status = response.status_code
            if return_status == 200:
                file_name_header = tck + "_" + str(Current_Year) + str(Current_Month).zfill(2) + str(day_cnt).zfill(2)
                print("Successful Request To TI ", file_name_header)
            else :
                print("Problems With Request To TI ", file_name_header)
                return SymCount
            soup = BeautifulSoup(response.content, 'html.parser')
            all_data = soup.find_all('tr')
            for tbl_row in all_data:
                if len(tbl_row) == 4:
                    fld_count = 0
                    fld_value = ''
                    for row_cell in tbl_row:
                        if fld_count == 0:
                            field_value = row_cell.text
                        if fld_count == 2:
                            field_name = row_cell.text
                        fld_count += 1
                    if Valid_Alert(Alert_Name_List, field_name) :
                        field_value = field_value.replace(",","")
                        new_rec = {'Ticker_Date' : file_name_header,
                                   'Alert_Filter_Name' : field_name,
                                   'Alert_Value' : field_value,
                                   'Filter_Value1' : '',
                                   'Filter_Value2' : ''
                                   }
                        alert_values = alert_values.append(new_rec, ignore_index=True)
                        print(file_name_header, field_name, " ,", field_value)
                if (len(tbl_row) > 4) :
                    fld_count = 0
                    field_value1 = ''
                    field_value2 = ''
                    for row_cell in tbl_row:
                        if fld_count == 0:
                            field_name = row_cell.text
                        if fld_count == 2:
                            field_value1 = row_cell.text
                        if fld_count == 3 :
                            tags_with_href = row_cell.find(href=True)
                            if tags_with_href == None :
                                field_value2 = row_cell.text
                        fld_count += 1
 #                   if Valid_Alert(Alert_Name_List, field_name):
                    field_value1 = field_value1.replace(",", "")
                    field_value2 = field_value2.replace(",", "")
                    new_rec = {'Ticker_Date': file_name_header,
                               'Alert_Filter_Name': field_name,
                               'Alert_Value': '',
                               'Filter_Value1': field_value1,
                               'Filter_Value2': field_value2
                               }
                    alert_values = alert_values.append(new_rec, ignore_index=True)
                    print(file_name_header, field_name, " ,", field_value1, field_value2)

            day_cnt = day_cnt - 1

        ticker_file_name = tck + "_" + str(Current_Year) + str(Current_Month).zfill(2) + ".csv"
        alert_values.to_csv(ticker_file_name, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
        SymCount = SymCount + 1
    return SymCount

# Check to see if this file is being executed as the "Main" python
# script instead of being used as a module by some other python script
# This allows us to use the module which ever way we want.
def main(argv):
   global ID_Counter
   inputfile = 'Ticker_List.csv'
   ID_Counter = 1
   outputdir = 'Dnld_Data'
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","odir="])
   except getopt.GetoptError:
      print('Trade_Ideas_Dnld -i <inputfile> -o <outputdir>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Trade_Ideas_Dnld -i <inputfile> -o <outputdir>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--odir"):
         outputdir = arg

   print('Input file is :', inputfile)
   print('Output directory is : ', outputdir)
   # Read in list of possible alerts
   Alert_Name_List = pd.read_csv('TI_Alert_Names.csv')
   return_code = Get_Data(Alert_Name_List, inputfile, outputdir)

   if return_code > 0 :
       print("Symbols Processed = ", return_code)
   else:
       print("No Symbols Processed From ",inputfile)
       return False

if __name__ == "__main__":
   main(sys.argv[1:])