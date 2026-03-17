import os 
import zipfile
from mega import Mega


#Work in Progress
#Supposed to be run when the data is needed locally to help keep the git usable
def megashit():
    #Object"
    mega_obj= Mega()

    #URL
    url= ""     #Input when data analysis is finsihed


    #Directory Management
    if os.path.exists("data"):
        print("You already ran this script bro")
        return 
    if not os.path.exists("data"):
        os.makedirs("data")
        zip_data= mega_obj.download_url(url)
        print("Downloading from MEGA")
        filename= str(zip_data)
        with zipfile.ZipFile(filename) as fn:
            print("Extracting data")
            fn.extractall("data")
        os.remove(filename)
        print("Done")
        return
if __name__ == "__main__":
    megashit()


#Hello Renee :)
