import requests
from bs4 import BeautifulSoup

ip = '10.14.73.253'

url = f"http://{ip}/"

response = requests.get('http://10.14.73.253')
soup = BeautifulSoup(response.content, "html.parser")
img_tags = soup.find_all("img")
for img_tag in img_tags:
    print("Found <img>:", img_tag)
# print(soup.prettify())