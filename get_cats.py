import requests
import os

# Make sure to replace 'your_api_key' with your actual API key
api_key = os.environ['CAT_API_KEY']
url = 'https://api.thecatapi.com/v1/images/search'

headers = {
    'x-api-key': api_key
}

# Create a directory to store the images
if not os.path.exists('cat_images'):
    os.makedirs('cat_images')

for i in range(9999, 100000):
    while True:
        try:
            response = requests.get(url, headers=headers).json()
            image_url = response[0]['url']
            image_data = requests.get(image_url).content
            with open('cats/' + str(i) + '.jpg', 'wb') as handler:
                handler.write(image_data)
            break
        except:
            print('Error')