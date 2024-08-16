import json
import urx
import robotiq_gripper

ip = "192.168.2.207"   #ip of the UR robot to connect

gripper = robotiq_gripper.RobotiqGripper()  # initialize the gripper
print("Connecting to gripper...")
gripper.connect(ip, 63352)                  # connect to the gripper
                                           
rob = urx.Robot(ip, use_rt=True)            # connect to the UR robot

############################################################################
manipulator_with_gripper = 
manipulator_with_tool = 
############################################################################

def new_product(product):  # add a new element in the json file with the position of the robot and gripper to grasp.
    x = rob.getl() # reads the current position of the robot
    name = product
    action = gripper.get_current_position()  # read the current position of the gripper
    with open('positions1.json') as json_file:  # open the json file with the previuos positions saved
        json_decoded = json.load(json_file)
    pos = {
        "action":action,
        "xref": (x[0]),
        "yref": (x[1]),
        "zref": (x[2]),
        "Rxref": (x[3]),
        "Ryref": (x[4]),
        "Rzref": (x[5])
        }

    counter = 2
    json_decoded[name] = pos

    print(len(json_decoded)-1)
    total = len(json_decoded)
    n = {"n": total-1}

    json_decoded['general'] = n

    with open('positions1.json', 'w') as json_file:  # save the new position
        json.dump(json_decoded, json_file, indent=2)

############################################################################

# def move_top(product):  # moves the robot TCP (tool central point) to the top of the product with the saved orientation
#     n_counter = product
#     with open('positions1.json', 'r') as fa:  # reads the position of the product saven in the json file with the especific name
#         data = json.load(fa)
#         n_points = data['general']['n']

#         action = data[str(n_counter)]['action']
#         x_refr = data[str(n_counter)]['xref']
#         y_refr = data[str(n_counter)]['yref']
#         z_refr = data[str(n_counter)]['zref']
#         ang_1 = data[str(n_counter)]['Rxref']
#         ang_2 = data[str(n_counter)]['Ryref']
#         ang_3 = data[str(n_counter)]['Rzref']

#     rob.movel((x_refr,y_refr, 0.45, ang_1, ang_2, ang_3),0.1, 0.1)  # send the position of the TCP to the top of the objec 45 cm on the top
    
def move_to_top_of_object(manipulator_name,object_name):  # moves the robot TCP (tool central point) to the top of the product (at approximately 45 cm on the top of the object position)
    n_counter = object_name
    with open('positions1.json', 'r') as fa:  # reads the position of the product saven in the json file with the especific name
        data = json.load(fa)
        n_points = data['general']['n']

        action = data[str(n_counter)]['action']
        x_refr = data[str(n_counter)]['xref']
        y_refr = data[str(n_counter)]['yref']
        z_refr = data[str(n_counter)]['zref']
        ang_1 = data[str(n_counter)]['Rxref']
        ang_2 = data[str(n_counter)]['Ryref']
        ang_3 = data[str(n_counter)]['Rzref']

    manipulator_name.movel((x_refr,y_refr, 0.45, ang_1, ang_2, ang_3),0.1, 0.1)  # send the position of the TCP to the top of the objec 45 cm on the top
       


############################################################################

# def move_grasp(product):  # moves the robot TCP (tool central point) to the the product as saved in the "new_product" function
#     n_counter = product
#     with open('positions1.json', 'r') as fa:
#         data = json.load(fa)
#         n_points = data['general']['n']

#         action = data[str(n_counter)]['action']
#         x_refr = data[str(n_counter)]['xref']
#         y_refr = data[str(n_counter)]['yref']
#         z_refr = data[str(n_counter)]['zref']
#         ang_1 = data[str(n_counter)]['Rxref']
#         ang_2 = data[str(n_counter)]['Ryref']
#         ang_3 = data[str(n_counter)]['Rzref']

#     rob.movel((x_refr,y_refr, z_refr, ang_1, ang_2, ang_3),0.1, 0.1)


def move_from_top_to_object(manipulator_name, object_name):  # moves the robot TCP (tool central point) from the top of the object to the position of the object

    n_counter = object_name
    with open('positions1.json', 'r') as fa:
        data = json.load(fa)
        n_points = data['general']['n']

        action = data[str(n_counter)]['action']
        x_refr = data[str(n_counter)]['xref']
        y_refr = data[str(n_counter)]['yref']
        z_refr = data[str(n_counter)]['zref']
        ang_1 = data[str(n_counter)]['Rxref']
        ang_2 = data[str(n_counter)]['Ryref']
        ang_3 = data[str(n_counter)]['Rzref']

    manipulator_name.movel((x_refr,y_refr, z_refr, ang_1, ang_2, ang_3),0.1, 0.1


############################################################################
# def close_gripper_on_object(manipulator_name,object_name):   # closes the gripper in order to grasp the object

# def grasp(product):   # closes the gripper according to the position dsaved in the json file. 
#     n_counter = product
#     with open('positions1.json', 'r') as fa:
#         data = json.load(fa)
#         n_points = data['general']['n']

#         action = data[str(n_counter)]['action']
#         x_refr = data[str(n_counter)]['xref']
#         y_refr = data[str(n_counter)]['yref']
#         z_refr = data[str(n_counter)]['zref']
#         ang_1 = data[str(n_counter)]['Rxref']
#         ang_2 = data[str(n_counter)]['Ryref']
#         ang_3 = data[str(n_counter)]['Rzref']

#     gripper.move_and_wait_for_pos(action,100,100)  # move the gripper to the saved position. 

def close_gripper_on_object(manipulator_name,object_name):   # closes the gripper in order to grasp the object
    n_counter = object_name
    with open('positions1.json', 'r') as fa:
        data = json.load(fa)
        n_points = data['general']['n']

        action = data[str(n_counter)]['action']
        x_refr = data[str(n_counter)]['xref']
        y_refr = data[str(n_counter)]['yref']
        z_refr = data[str(n_counter)]['zref']
        ang_1 = data[str(n_counter)]['Rxref']
        ang_2 = data[str(n_counter)]['Ryref']
        ang_3 = data[str(n_counter)]['Rzref']

    gripper_name = manipulator_name + "_gripper" 
    gripper_dict[gripper_name].move_and_wait_for_pos(action,100,100)  # move the gripper to the saved position. 

############################################################################

# def gripper_open(manipulator_name):  # open the gripper completelly 
#     gripper.move_and_wait_for_pos(0,100,100)

def open_gripper(manipulator_name):  # open the gripper completelly 
    gripper_name = manipulator_name + "_gripper" 
    gripper_dict[gripper_name].move_and_wait_for_pos(0,100,100)


############################################################################
def return_to_rest_position(manipulator_name): # moves the robot TCP (tool central point) to the manipulators rest position



    




# def move_to_top_of_object(manipulator_name,object_name):  # moves the robot TCP (tool central point) to the top of the product (at approximately 45 cm on the top of the object position)

# def move_from_top_to_object(manipulator_name, object_name):  # moves the robot TCP (tool central point) from the top of the object to the position of the object

# def close_gripper_on_object(manipulator_name,object_name):   # closes the gripper in order to grasp the object

# def open_gripper(manipulator_name):  # open the gripper completelly 

# def return_to_rest_position(manipulator_name): # moves the robot TCP (tool central point) to the manipulators rest position

# def Apply_force(manipulator_name) #Apply a force in the vertical direction mainly for cutting objects


    