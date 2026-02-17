# %%
import numpy as np
import torch.nn as nn
import torch
import cv2
import cnn 
import ImageProcessing as ip
import sympy
import pandas as pd
import os


# %%
# Instantierer 3 CNN's
mnist_net = cnn.mnist_layers()
math_net = cnn.math_layers()
classifier_net = cnn.classifier_layers()

# Indlæser vægtene og bias fra forrigt trænede CNN
mnist_net.load_state_dict(torch.load("weights/mnist/ep36loss0.036657acc0.989050", weights_only=True))
math_net.load_state_dict(torch.load("weights/math/ep38loss0.001249acc0.999700", weights_only=True))
classifier_net.load_state_dict(torch.load("weights/classifier/ep39loss0.000048acc1.000000", weights_only=True))

# Sæt modellerne i evalueringsmodus, således den ikke bruger teknikker, der kun anvendes under træning
mnist_net.eval()  
math_net.eval() 
classifier_net.eval()

print("Model indlæst og klar til brug")
sm = nn.Softmax(dim=1)

# %%
# https://www.google.com/search?q=digit+recognition+on+video+python&sca_esv=4a2852a2c4361dc0&sxsrf=AHTn8zqYWdEubWjBZQeWokwTbpJR22F9Ig%3A1737714687230&ei=_2uTZ5fUDZLWwPAP2ZvY8AE&ved=0ahUKEwjXz5zVk46LAxUSKxAIHdkNFh4Q4dUDCBA&uact=5&oq=digit+recognition+on+video+python&gs_lp=Egxnd3Mtd2l6LXNlcnAiIWRpZ2l0IHJlY29nbml0aW9uIG9uIHZpZGVvIHB5dGhvbjIFEAAY7wUyBRAAGO8FMgUQABjvBTIFEAAY7wUyBRAAGO8FSKM4UI8FWL0vcAJ4AZABAJgBjwGgAfwRqgEEMy4xN7gBA8gBAPgBAZgCE6ACyA_CAgoQABiwAxjWBBhHwgIKECEYoAEYwwQYCsICBBAhGAqYAwCIBgGQBgiSBwQ0LjE1oAe9XA&sclient=gws-wiz-serp#fpstate=ive&vld=cid:785cccfb,vid:jBwOFjtH89U,st:0
# Laver kamera objekt
cam = cv2.VideoCapture(0)

# Henter default frame bredde og højde
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Laver box i midten af frame, som skal aflæse tallet
box_size = (200*2, 200) # Definerer tupel der aagiver størrelsen på boxen
box = [(int(frame_width // 2 - box_size[0] // 2), int(frame_height // 2 - box_size[1] // 2)),
       (int(frame_width // 2 + box_size[0] // 2), int(frame_height // 2 + box_size[1] // 2))] # Angiver boxens koordinater centreret i framen i en liste af tupels

# %%
def to_model_tensor(image):
    image = torch.Tensor(image)
    image = image.float()/255
    image = image.unsqueeze(0).unsqueeze(0)
    return image

# %%
def calculate(pred):
    try:
        result = sympy.sympify(pred)
        return result 
    except sympy.SympifyError as e:
        return "Cannot be calculated"


# Laver varibler og funktion, der kan gemme resultater til eksperiment
frame_count = 0
count_limit = 10
results = []
sample_data = []
label = ""
space_bar_pressed = False
count = 0

def save_results(results, label):
    right_answers = 0

    for result in results:
        result = result.strip()
        label = label.strip()
        right_digits = 0
        min_length = min(len(result), len(label))

        for i in range(min_length):
            if result[i] == label[i]:
                right_digits += 1

        equation_accuracy = right_digits / min_length

        right_answers += equation_accuracy

    return right_answers / len(results)


# %%
while True:
    # Læser fra kameraet og viser dette i frame
    ret, frame = cam.read() 
    frame = cv2.flip(frame, 1) # Spejlvender kameraet

    # Tager udsnit af framen til boxen, markerer en box med tidligere definerede koordinater og viser i nyt vindue 
    cropped_frame = frame[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    cropped_frame = cv2.flip(cropped_frame, 1) # Spejlvender kameraet
    box_vid = ip.preprocess_stack(cropped_frame)

    # Henter modellens genkendelse af tallet 
    image = ip.preprocess_stack(cropped_frame)
    digits, bounding_boxes = ip.seperate_digits(image)
    preds = []
    display_images = []

    for idx, d in enumerate(digits):
        # Hvis bounding box'en faktisk har en størrelse gøres følgende
        if d.size:
            d = cv2.resize(d, (28,28))
            display_images.append(cv2.resize(d, (112,112)))
            d = to_model_tensor(d)
                
            classifier_pred = sm(classifier_net(d))
            classifier_pred_class = classifier_pred.argmax().item()
            classifier_prob = classifier_pred[0][classifier_pred_class].item()

            if classifier_prob < 0.8:    
                mnist_pred = sm(mnist_net(d))
                mnist_pred_digit = mnist_pred.argmax().item()
                mnist_prop = mnist_pred[0][mnist_pred_digit].item()

                math_pred = sm(math_net(d))
                math_pred_digit = math_pred.argmax().item()                
                math_prop = math_pred[0][math_pred_digit].item()    
            
                if math_pred_digit == 0:
                    math_pred_digit = '+'
                elif math_pred_digit == 1:
                    math_pred_digit = '-'
                elif math_pred_digit == 2:
                    math_pred_digit = '*'
                if math_prop > mnist_prop:
                    preds.append((math_pred_digit, math_prop))
                else:
                    preds.append((mnist_pred_digit, mnist_prop))


            # Hvis objektet er et tal
            elif classifier_pred_class == 0:
                mnist_pred = sm(mnist_net(d))
                mnist_pred_digit = mnist_pred.argmax().item()
                mnist_prop = mnist_pred[0][mnist_pred_digit].item()
                preds.append((mnist_pred_digit, mnist_prop))

            # Hvis objektet er et matematisk symbol
            elif classifier_pred_class == 1:
                math_pred = sm(math_net(d))
                math_pred_digit = math_pred.argmax().item()                
                math_prop = math_pred[0][math_pred_digit].item()    
                
                if math_pred_digit == 0:
                    math_pred_digit = '+'
                elif math_pred_digit == 1:
                    math_pred_digit = '-'
                elif math_pred_digit == 2:
                    math_pred_digit = '*'

                preds.append((math_pred_digit, math_prop))
    
    pred_digits = [pred[0] for pred in preds]
    pred_string = ''.join(map(str, pred_digits))

    for bbox in bounding_boxes:
        (x, y, w, h) = bbox
        bbox_start = (x,y)
        bbox_end = (x + w, y + h)
        cv2.rectangle(box_vid, bbox_start, bbox_end, (255, 0, 0), 3)


    cv2.putText(frame, f"Prediction: {pred_string} = {calculate(pred_string)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Label: {label}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Observation Count: {count}", (50,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    if space_bar_pressed:
        cv2.putText(frame, f"SAVED RESULTS", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(box_vid, f"Digit count: {len(digits)}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    
    cv2.rectangle(frame, box[0], box[1], (0, 0, 0), 3)


    key = cv2.waitKey(1)


    experiment_count = 4
    abs_path = f"model_images/exp{experiment_count}"


    # Gemmer resultater fra 10 frames ved at trykke på 'space'-knappen
    if key == 32 and frame_count == 0:
        frame_count = count_limit 
        results = []
        space_bar_pressed = True
        os.mkdir(f"{abs_path}/obs{count}")
        observation_path = f"{abs_path}/obs{count}"

        cv2.imwrite(f"{observation_path}/frame.jpg", cropped_frame)
        cv2.imwrite(f"{observation_path}/frame_proccesed.jpg", box_vid)

        for idx, bbox in enumerate(display_images):
            cv2.imwrite(f"{observation_path}/bbox{idx}.jpg", bbox)

    if len(display_images):
        display_images = np.concatenate(display_images, axis=1)
        cv2.imshow('Model input', display_images)

 
    if frame_count > 0:
        results.append(pred_string) 
        frame_count -= 1 

        if frame_count == 0:
            accuracy = save_results(results, label)
            sample_data.append([accuracy, label, results])
            space_bar_pressed = False
            count += 1
            print("Accuracy:", accuracy, "\nLabel:", label, "\nResults:",results, )
            print("-------------------------------------------------------------------------------------------------\n") 
            

    # Lukker kamera-vinduet ved at trykke på 'esc'-knappen
    if key == 27:
        break

    # Sletter tegn i label ved at trykke på 'back-space'-knappen
    # ChatGPT
    elif key == 127:
        if len(label) > 0:
            label = label[:-1]

    # Tilføjer tegn til label
    elif 33 <= key <= 126:  
        label += chr(key)

    cv2.imshow('Debug', box_vid)
    cv2.imshow('Camera with Prediction', frame)

# Save sample_data in excel file
file_path = 'Experiment_data.xlsx'
sheet_name = 'exp_5'

df = pd.DataFrame(sample_data, columns=['Accuracy', "Label", "Sample-data"])

if not os.path.exists(file_path):
    df.to_excel(file_path, sheet_name=sheet_name, index=False)
    print(f"File {file_path} created with sheet '{sheet_name}'!")
else:
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Sheet '{sheet_name}' added to the existing file {file_path}!")

cv2.destroyAllWindows()

