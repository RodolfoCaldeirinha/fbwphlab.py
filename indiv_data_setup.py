import os 
import zipfile
import urllib.request


#Work in Progress
#Supposed to be run when the data is needed locally to help keep the git usable
def megashit():
    #Object"
   

    #URL
    url= "https://www.dropbox.com/scl/fi/offbbw8hdecn8rb4opmz2/CSV-data_processed.zip?rlkey=8lgp6iwqcixrsdbx7bu6kzfqh&st=2pqvwpr5&dl=1"     #Input when data analysis is finsihed


    #Directory Management
    if os.path.exists("data"):
        print("You already ran this script bro")
        return 
    if not os.path.exists("data"):
        os.makedirs("data")
        

        filename = "temp_data.zip" 
        urllib.request.urlretrieve(url, filename)
        print("Downloading from Dropbox since Mega is not the brightest")
        with zipfile.ZipFile(filename) as fn:
            print("Extracting data")
            fn.extractall("data")
        os.remove(filename)
        print("Done")
        return
if __name__ == "__main__":
    megashit()


#Hello Renee :)
