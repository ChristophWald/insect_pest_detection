def sliding_window_prediction(image_path, model, window_size=(640, 640), stride=640, conf_threshold=0.0, visual_aid = False, save = False):
    # Load the image
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    #print("Original picture: ", height, width)

    # Get the window size and stride (non-overlapping, so stride = window_size)
    window_height, window_width = window_size
    num_windows_y = (height - window_height) // stride + 1
    num_windows_x = (width - window_width) // stride + 1

    #print("Number of vertical(y) and horizontal(x) windows: ", num_windows_y, num_windows_x)

    tiles = []
    # Initialize lists to store confidence, class, and bounding box data
    confidences = []
    classes = []
    boxes = []
    n_objects = []
    
    object_count = 0
    for i in range(num_windows_y):
            for j in range(num_windows_x):
                # Define the window's coordinates
                y_start = i * stride
                y_end = y_start + window_height
                x_start = j * stride
                x_end = x_start + window_width
        
                # Extract the window (640x640 tile)
                window = img[y_start:y_end, x_start:x_end]
                #print(y_start, y_end, x_start, x_end)
                
                # Zero-pad the window to the target size (640x640) if it's smaller
                pad_top = max(0, 0 - y_start)  # Padding on the top
                pad_bottom = max(0, (y_end - height))  # Padding on the bottom
                pad_left = max(0, 0 - x_start)  # Padding on the left
                pad_right = max(0, (x_end - width))  # Padding on the right
    
                # Apply zero-padding using cv2.copyMakeBorder
                window_padded = cv2.copyMakeBorder(window, pad_top, pad_bottom, pad_left, pad_right, 
                                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))

                window_rgb = cv2.cvtColor(window_padded, cv2.COLOR_BGR2RGB)  # Convert to RGB
                window_normalized = window_rgb / 255.0  # Normalize to [0, 1]
                window_tensor = torch.tensor(window_normalized).permute(2, 0, 1).unsqueeze(0).float().to(model.device)
                
                # Make the prediction
                results = model(window_tensor, verbose=False, augment = True)
                # Filter the results by confidence score threshold
                predictions = results[0].boxes
                confidences_pred = predictions.conf
                valid_predictions = predictions[confidences_pred > conf_threshold]
    
                # Check if there are valid predictions
                if len(valid_predictions) > 0:
                    object_count += len(valid_predictions)
                    n_objects.append(len(valid_predictions))
                    tiles.append([y_start, y_end, x_start, x_end])  # Store tile coordinates
                    #print(f"Objects detected in window ({y_start, y_end}, {x_start, x_end}) - Count: {len(valid_predictions)}")
                    
                    #Create a copy of the window to draw the bounding boxes
                    window_with_bboxes = window_padded.copy()
                    for idx, (bbox, cls, conf) in enumerate(zip(valid_predictions.xyxy.cpu().numpy(), valid_predictions.cls.cpu().numpy(), valid_predictions.conf.cpu().numpy())):
                        xmin, ymin, xmax, ymax = bbox
                        # Draw the bounding box for each valid prediction
                        #print(xmin, ymin, xmax, ymax)
                        cv2.rectangle(window_with_bboxes, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                        label = f"Class {int(cls)}: {conf:0.2f}"  # Label with class id and confidence
                        cv2.putText(window_with_bboxes, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
                        # Save the bounding box for later (in a proper format)
                        boxes.append([xmin, ymin, xmax, ymax])
                        confidences.append(conf)
                        classes.append(int(cls))

                        if visual_aid:
                            plt.imshow(window_with_bboxes)
                            plt.show()
                        if save:
                            cv2.imwrite(f"tile_with_bboxes_{y_start}_{x_start}.jpg", window_with_bboxes)
                # Convert the lists to numpy arrays for easy processing later

    boxes = np.array(boxes)
    confidences = np.array(confidences)
    classes = np.array(classes)
            
    print(f"Total objects detected: {object_count}")
    cv2.destroyAllWindows()  # Close all OpenCV windows after finishing the loop
    return object_count, n_objects, tiles, confidences, classes, boxes

def plot_bboxes(image_path, confidences, classes, boxes, tiles, object_n):
    # Load the original image
    img = cv2.imread(image_path)

    bbox_idx = 0  # To track the current index in the sorted bounding boxes list

    # Loop over all the tiles
    for tile_idx, (y_start, y_end, x_start, x_end) in enumerate(tiles):
        # Get the number of objects (bounding boxes) in the current tile from object_n
        n_objects_in_tile = object_n[tile_idx]

        # Loop over the number of bounding boxes in the current tile
        for i in range(n_objects_in_tile):
            # Get the bounding box, class, and score for the current object
            score = confidences[bbox_idx]
            cls = classes[bbox_idx]
            bbox = boxes[bbox_idx]
            bbox_idx += 1  # Move to the next bounding box

            # Get the coordinates of the bounding box
            xmin, ymin, xmax, ymax = bbox

            # Adjust the bounding box coordinates to the full image (tile offset)
            xmin_full = int(xmin + x_start)
            ymin_full = int(ymin + y_start)
            xmax_full = int(xmax + x_start)
            ymax_full = int(ymax + y_start)

            # Get the class label
            label = f"Class {int(cls)}: {score:0.2f}"  # Create the label text
            lbl_margin = 3  # Label margin

            # Draw the bounding box on the image
            img = cv2.rectangle(img, (xmin_full, ymin_full),
                                (xmax_full, ymax_full),
                                color=(0, 0, 255),
                                thickness=2)

            # Calculate the label size and add margins
            label_size = cv2.getTextSize(label,  # label size in pixels
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1, thickness=1)
            lbl_w, lbl_h = label_size[0]  # label width and height
            lbl_w += 2 * lbl_margin  # add margins on both sides
            lbl_h += 2 * lbl_margin

            # Draw the label background (filled rectangle)
            img = cv2.rectangle(img, (xmin_full, ymin_full - lbl_h),  # top-left
                                 (xmin_full + lbl_w, ymin_full),  # bottom-right
                                 color=(0, 0, 255),
                                 thickness=-1)  # thickness=-1 means filled rectangle

            # Write the label text to the image
            cv2.putText(img, label, (xmin_full + lbl_margin, ymin_full - lbl_margin),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0, color=(255, 255, 255),
                        thickness=1)

    return img


model = YOLO('/user/christoph.wald/u15287/06_check_lables/runs/detect/train7/weights/best.pt')
image_path = "/user/christoph.wald/u15287/05_praktikum_final/data_raw/train_BRAIIM0003.jpg"
image_path = "/user/christoph.wald/u15287/06_check_lables/images/0c595bdb-429_1SCIAF_CanonEOS_woal.JPG"
print(image_path)
_ , n_objects, tiles, confidences, classes, boxes = sliding_window_prediction(image_path, model, conf_threshold=0.5)
img = plot_bboxes (image_path, confidences, classes, boxes, tiles,n_objects)
show(img)