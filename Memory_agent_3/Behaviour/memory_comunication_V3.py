import requests

# Function to add information
def add_info(flag, info_to_add):
    url = "http://192.168.2.185:8004/addinfo"  # Endpoint URL
    data = {"info": info_to_add, "flag" : flag}  # Request payload
    response = requests.post(url, json=data)  # Send POST request
    if response.status_code == 200:
        print("Info added successfully")
    else:
        print("Failed to add info")


# Function to retrieve Answer
def retrieve_answer(user_text_request):
    url = "http://192.168.2.185:8004/retrieveAnswer"  # Endpoint URL
    data = {"text": user_text_request}  # Request payload
    response = requests.post(url, json=data)  # Send POST request
    if response.status_code == 200:
        return response.text
    else:
        return "Failed to retrieve info"
    
    

# Function to retrieve Behaviour
def retrieve_behaviour(user_text_request):
    url = "http://192.168.2.185:8005/retrieveBehaviour"  # Endpoint URL
    data = {"text": user_text_request}  # Request payload
    response = requests.post(url, json=data)  # Send POST request
    if response.status_code == 200:
        return response.text
    else:
        return "Failed to retrieve info"
    
    
    
# Function to retrieve Law
def retrieve_laws(user_text_request):
    url = "http://192.168.2.185:8006/retrieveLaw"  # Endpoint URL
    data = {"text": user_text_request}  # Request payload
    response = requests.post(url, json=data)  # Send POST request
    if response.status_code == 200:
        return response.text
    else:
        return "Failed to retrieve info"


##########################################################################################
#     Usage example (Add at least three information to the memory at the beginning)
##########################################################################################


##################  Add Info to memory collection ##################

add_info("ANS","The cola is in the kitchen")
add_info("ANS","The robot manipulator is working")
add_info("ANS","The apple is red")

###################  Add Info to Behaviour collection  ##############

info_to_add = ''' 
"task": "find object by name",
"behavior_pattern": ">>>QUESTION_VIEW(Is there any <object_name>?), <<<RESULT(yes), >>>SEARCH_VIEW(<object_name>), <<<RESULT(<p><object_name></p>)"
'''
add_info("BEH",info_to_add)

info_to_add = ''' 
"task": "find object by class",
"behavior_pattern": ">>>QUESTION_VIEW(Is there any food?), <<<RESULT(yes), >>>QUESTION_VIEW(what food is there?), <<<RESULT(yes), >>>SEARCH_VIEW(<object_name>), <<<RESULT(<p><object_name></p>), >>>GO_TO(<p><object_name></p>)"
'''
add_info("BEH",info_to_add)

info_to_add = ''' 
"task": "solve mathematical problem",
"behavior_pattern": ">>>THOUGHT(the sum of 2 and 5 is 7), <<<RESULT(success), >>>SAY(The result is 7), <<<RESULT(success)"
'''
add_info("BEH",info_to_add)


###################  Add Info to Law collection  ##################

info_to_add = ''' 
Law 1: Avoid causing harm to humans,
description: A robot may not injure a human being or allow a human being to come to harm.
'''
add_info("LAW",info_to_add)


info_to_add = ''' 
Law 2: Follow orders from humans, unless it conflicts with Law 1,
description: A robot must obey orders given by humans, except where it conflicts with the first law.
'''
add_info("LAW",info_to_add)


info_to_add = ''' 
Law 3: Ensure self-preservation without violating the first two laws.,
description: A robot must protect its own existence as long as it doesn't conflict with the first two laws.
'''
add_info("LAW",info_to_add)




print("All saved")





