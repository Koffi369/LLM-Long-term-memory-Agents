import requests

# Function to add information
def add_info(info_to_add):
    url = "http://192.168.2.185:8004/addinfo"  # Endpoint URL
    data = {"info": info_to_add}  # Request payload
    response = requests.post(url, data=data)  # Send POST request
    if response.status_code == 200:
        print("Info added successfully")
    else:
        print("Failed to add info")

# Function to retrieve information
def retrieve_info(user_text_request):
    url = "http://192.168.2.185:8004/retrieveinfo"  # Endpoint URL
    data = {"text": user_text_request}  # Request payload
    response = requests.post(url, data=data)  # Send POST request
    if response.status_code == 200:
        return response.text
    else:
        return "Failed to retrieve info"

# Usage example
add_info("Some information to add")
response = retrieve_info("User's text request")
print(response)