from astropy.table import Table


##Download files:
import requests
import os
from astropy.io import fits
def download_file(url, file_path, ID):
    if os.path.exists(file_path):
        print(f"File already exists for ID {ID}, skipping download.")
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print("Download successful for ID %s."%(ID))
    else:
        print("Failed to download ID %s."%(ID))

def download_all(inputfile,field='fresco-only-n-v2',save_path ='./rows_phots/'):
    idlist=inputfile['id']
    for i,ID in enumerate(idlist):
        ID = int(ID)
        # File URL
        filename=field+"_"+str(ID).zfill(5)+".stack.fits"
        urlbeams = "https://s3.amazonaws.com/grizli-v2/HST/Pipeline/"+field+"/Extractions/"+filename
        download_file(urlbeams, save_path+filename,ID)
        print('---')

data = Table.read("C:\\Users\\casey\\Desktop\\fresco-gn_imgv7.3_photcatv1_swlw_fo_det_aper8_zphot.cat.fits")
download_all(data, field='fresco-only-n-v2', save_path='C:\\Users\\casey\\Desktop\\rows_phots\\')

"""
For Google Colab

from astropy.table import Table


##Download files:
import requests
import os
from astropy.io import fits
def download_file(url, file_path, ID):
    if os.path.exists(file_path):
        print(f"File already exists for ID {ID}, skipping download.")
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print("Download successful for ID %s."%(ID))
    else:
        print("Failed to download ID %s."%(ID))

def download_all(inputfile,field='fresco-only-n-v2',save_path ='./rows_gn/'):
    idlist=inputfile['id']
    for i,ID in enumerate(idlist):
        ID = int(ID)
        # File URL
        filename=field+"_"+str(ID).zfill(5)+".stack.fits"
        urlbeams = "https://s3.amazonaws.com/grizli-v2/HST/Pipeline/"+field+"/Extractions/"+filename
        download_file(urlbeams, save_path+filename,ID)
        print('---')

data = Table.read("/content/FRESCOCAT_GN_prelim.csv", format='csv')
os.makedirs("/content/rows_gn", exist_ok=True)
download_all(
    data[(data['Quality_Alba'] >= 2) * (data['Quality_Alba'] < 4)],
    field='fresco-only-n-v2',
    save_path='/content/rows_gn/'
)
"""