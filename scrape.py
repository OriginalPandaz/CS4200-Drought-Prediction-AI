import requests, csv
from bs4 import BeautifulSoup

def get_dataset(filename):

    linkAddress = 'http://www.laalmanac.com/weather/we08aa.php'
    address = requests.get(linkAddress) #Get HTML from link
    soup = BeautifulSoup(address.content, 'html.parser') #Parse the HTML

    dataset = soup.find('table') #Get strictly dataset

    with open(filename, mode='w') as table:
        for data in dataset.find_all('tbody'): #Search each instance of the dataset
            text = data.find_all('tr')
            stats = [x.text.strip() for x in text] #Acquire values from HTML]
            write = csv.writer(table) #Setup writer for csv
            write.writerow(stats) #Write to csv

get_dataset('precip.csv')
