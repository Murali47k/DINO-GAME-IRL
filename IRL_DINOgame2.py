import cv2
import numpy as np
import random
import os
import mediapipe as md
import time
from math import sqrt

gameOver = False
score = 0
x_enemy = 0
y_enemy1 = 200 # random.randint(0, 150)
y_enemy2 = 335 # random.randint(300, 450)
y_enemy3 = 335 # random.randint(300, 450)
enemynow = y_enemy2

md_drawing=md.solutions.drawing_utils
md_drawing_styles=md.solutions.drawing_styles
md_pose=md.solutions.pose

# Path to the PNG image file
image_path1 = r'mini projects\IRL_DINO\enemy_images\enemyfly1.png'

image_path2 = r"mini projects\IRL_DINO\enemy_images\cactus1.png"

image_path3 = r"mini projects\IRL_DINO\enemy_images\cactus2.png"


# Check if the file exists
if not os.path.exists(image_path1):
    print(f"Error: The file '{image_path1}' does not exist.")
else:
    # Load the image
    png_image1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)

    # Check if the image was loaded correctly
    if png_image1 is None:
        print(f"Error: The image '{image_path1}' could not be loaded. Check the file path and integrity.")
    else:
        # Define desired dimensions for resizing
        desired_width, desired_height = 100,100  # Example dimensions
        
        # Resize the image
        resized_png = cv2.resize(png_image1, (desired_width, desired_height))

# Check if the file exists
if not os.path.exists(image_path2):
    print(f"Error: The file '{image_path2}' does not exist.")
else:
    # Load the image
    png_image2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)

    # Check if the image was loaded correctly
    if png_image2 is None:
        print(f"Error: The image '{image_path2}' could not be loaded. Check the file path and integrity.")
    else:
        # Define desired dimensions for resizing
        desired_width, desired_height = 80,80  # Example dimensions
        
        # Resize the image
        resized_png2 = cv2.resize(png_image2, (desired_width, desired_height))

# Check if the file exists
if not os.path.exists(image_path3):
    print(f"Error: The file '{image_path3}' does not exist.")
else:
    # Load the image
    png_image3 = cv2.imread(image_path3, cv2.IMREAD_UNCHANGED)

    # Check if the image was loaded correctly
    if png_image3 is None:
        print(f"Error: The image '{image_path3}' could not be loaded. Check the file path and integrity.")
    else:
        # Define desired dimensions for resizing
        desired_width, desired_height = 100,100  # Example dimensions
        
        # Resize the image
        resized_png3= cv2.resize(png_image3, (desired_width, desired_height))

# Function to overlay PNG with transparency
def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape

    # Ensure the coordinates and size are within the bounds of the background
    if x >= background.shape[1] or y >= background.shape[0]:
        return
    if x + w > background.shape[1]:
        w = background.shape[1] - x
    if y + h > background.shape[0]:
        h = background.shape[0] - y

    overlay = overlay[:h, :w]

    alpha_overlay = overlay[:, :, 2] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                                       alpha_background * background[y:y+h, x:x+w, c])

def enemy(frame):
    global score, x_enemy, y_enemy1,y_enemy2,y_enemy3,enemynow
    if x_enemy > imageWidth:  # Reset enemy position if it goes beyond the screen
        x_enemy = 0
        y_enemy1 = 200   #random.randint(50, 150)
        y_enemy2 =  335   #random.randint(320, 340)
        y_enemy3 =  335   #random.randint(320, 340)
        y_enemylst=[y_enemy1,y_enemy2,y_enemy3]
        enemynow = random.choice(y_enemylst)
        score+=1

    y_enemylst=[y_enemy1,y_enemy2,y_enemy3]
    enemydict={y_enemy1:resized_png,y_enemy2:resized_png2,y_enemy3:resized_png3}
    overlay_image(frame, enemydict[enemynow], x_enemy,  enemynow)

    x_speed = 10  # Adjust this value to control the speed of the enemy circle
    x_enemy += x_speed

video = cv2.VideoCapture(0)
with md_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while video.isOpened():


        ret, frame = video.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        imageHeight, imageWidth, _ = image.shape

        endimage = np.zeros(image.shape,np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Score", (480, 30), font, 1, (255, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, str(score), (590, 30), font, 1, (255, 0, 0), 4, cv2.LINE_AA)

        enemy(image)

        image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
        result=pose.process(image) # using model to identify in video

        lmList=[] # it will contain the position of all 32 points in the pos model

        if result.pose_landmarks: # if models is present
            md_drawing.draw_landmarks(image,result.pose_landmarks,md_pose.POSE_CONNECTIONS)

            for id,lm in enumerate(result.pose_landmarks.landmark):
                # id=no id of the point and im (landmark) is the x and y ratio value of point
                h,w,_=image.shape # height and width
                x,y=int(lm.x*w),int(lm.y*h)
                lmList.append([id,x,y])
        if len(lmList) != 0:

            _,top_left_corner_x,top_left_corner_y = lmList[11]
            _,bottom_right_corner_x,bottom_right_corner_y = lmList[24]
            _,top_right_corner_x,top_right_corner_y = lmList[12]
            _,bottom_left_corner_x,bottom_left_corner_y = lmList[23]
            enemy_pos_x = x_enemy+50
            enemy_pos_y1 = y_enemy1 + 50
            enemy_pos_y2 = y_enemy2 + 50 # same for y_enemy3

            cv2.rectangle(image,(top_left_corner_x,top_left_corner_y) ,(bottom_right_corner_x,bottom_right_corner_y),(0,0,255), 5)

            Distance1 = sqrt((top_left_corner_x-enemy_pos_x)**2 + (top_left_corner_y-enemy_pos_y1)**2) # using distance formula
            Distance2 = sqrt((top_right_corner_x-enemy_pos_x)**2 + (top_right_corner_y-enemy_pos_y1)**2)

            Distance3 = sqrt((bottom_left_corner_x-enemy_pos_x)**2 + (bottom_left_corner_y-enemy_pos_y2)**2)
            Distance4 = sqrt((bottom_right_corner_x-enemy_pos_x)**2 + (bottom_right_corner_y-enemy_pos_y2)**2)

            if enemynow == y_enemy1:
                if Distance1 <= 25 or Distance2 <= 25:
                    endscore = score
                    gameOver=True
                    
                    
            elif enemynow == y_enemy2 or enemynow == y_enemy3:
                if Distance3 <= 25 or Distance4 <= 25:
                    endscore = score
                    gameOver = True
                    

           
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        image = cv2.flip(image,1)

        text=cv2.putText(image,"Score",(480,30),font,1,(0,255,0),4,cv2.LINE_AA)
        text=cv2.putText(image,str(score),(590,30),font,1,(0,255,0),4,cv2.LINE_AA)

        if gameOver:
            text = cv2.putText(endimage,"GAME OVER",(100,200),font,2.5,(0,0,255),6,cv2.LINE_AA)
            text=cv2.putText(endimage,"Score :",(170,280),font,2,(0,0,255),5,cv2.LINE_AA)
            text=cv2.putText(endimage,str(endscore),(415,280),font,2,(0,0,255),5,cv2.LINE_AA)
            text = cv2.putText(endimage,"press Q to exit",(190,400),font,1,(0,0,255),3,cv2.LINE_AA)
            cv2.imshow('IRL-DINOgame', endimage)
        else:
            cv2.imshow('IRL-DINOgame', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
