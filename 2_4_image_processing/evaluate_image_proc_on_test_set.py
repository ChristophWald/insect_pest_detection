import sys
sys.path.append("/user/christoph.wald/u15287/insect_pest_detection/modules")
from modules_prediction import compare_labels_vectorized
from modules import load_yolo_labels
from modules_evaluation import compute_metrics, save_results_to_json

def parse_coords(line):
    line = line.strip()
    if not line:
        return None
    # Remove surrounding brackets or parentheses
    line = line.strip("[]()")
    # Split by comma and convert each element to int after stripping whitespace
    coords = [int(x.strip()) for x in line.split(",")]
    return coords

def evaluate_on_test_set_from_txt(save_images=False, save_results=True, skip_FRANOC=True):
    import time, os, json, torch, cv2
    start = time.time()
    results = []

    print(f"Evaluating predictions from image segmented labels on test set.")

    base_output_path = f"/user/christoph.wald/u15287/insect_pest_detection/2_5_self_training/metrics/image_procv2"
    os.makedirs(base_output_path, exist_ok=True)

    #images are rotated, so original test set images&labels would be wrong
    base_image_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/04_images_cropped"
    base_label_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/04_labels_cropped"
    base_pred_path = "/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/05_created_labels"

    #only the images of the test set that could be segmented are considered
    filenames = os.listdir("/user/christoph.wald/u15287/big-scratch/02_splitted_data/test_set/SSL/03_images_masked")
    filenames.sort()

    if save_images:
        image_output_path = os.path.join(base_output_path, "images_w_bboxes")
        os.makedirs(image_output_path, exist_ok=True)

    for filename in filenames:
        if skip_FRANOC and filename.startswith("FRANOC"):
            continue        
        

        # Load ground-truth labels, they are in (x,y,w,h) format (inconsistent to the [] in the segmented labesl)
        label_path = os.path.join(base_label_path, os.path.splitext(filename)[0] + ".txt")
        label_boxes = []
        with open(label_path, "r") as f:
              for line in f:
                coords = parse_coords(line)
                if coords is not None:
                    label_boxes.append(coords)

        if len(label_boxes) == 0:
            label_boxes = torch.empty((0,4), dtype=torch.float32).to("cuda")
            label_classes = torch.empty((0,), dtype=torch.long).to("cuda")
        else:
            label_boxes = torch.tensor(label_boxes, dtype=torch.float32).to("cuda")
            # Convert x,y,w,h -> x1,y1,x2,y2
            x1 = label_boxes[:,0]
            y1 = label_boxes[:,1]
            x2 = x1 + label_boxes[:,2]
            y2 = y1 + label_boxes[:,3]
            label_boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # Derive class ID from filename or from content if available
            species = ["BRAIIM", "LIRIBO","FRANOC", "TRIAVA"]
            row_index = next((i for i, sp in enumerate(species) if filename.startswith(sp)), len(species))
            label_classes = torch.tensor([row_index]*len(label_boxes), dtype=torch.long).to("cuda")


        # Load segmented labels, they are in [x,y,w,h] format
        pred_path = os.path.join(base_pred_path, os.path.splitext(filename)[0] + ".txt")
        pred_boxes = []
        with open(pred_path, "r") as f:
            for line in f:
                coords = parse_coords(line)
                if coords is not None:
                    pred_boxes.append(coords)


        if len(pred_boxes) == 0:
            pred_boxes = torch.empty((0,4)).to("cuda")
            pred_classes = torch.empty((0,), dtype=torch.long).to("cuda")
        else:
            pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32).to("cuda")
            '''     
            x1 = pred_boxes[:,0]
            y1 = pred_boxes[:,1]
            x2 = x1 + pred_boxes[:,2]
            y2 = y1 + pred_boxes[:,3]
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            '''
            # Derive class ID from filename
            species = ["BRAIIM", "LIRIBO","FRANOC", "TRIAVA"]  # adjust as needed
            row_index = next((i for i, sp in enumerate(species) if filename.startswith(sp)), len(species))
            pred_classes = torch.tensor([row_index]*len(pred_boxes), dtype=torch.long).to("cuda")
   
        print(f"pred_boxes {pred_boxes}")
        print(f"pred_classes {pred_classes}")
        print(f"label_boxes {label_boxes}")
        print(f"label_classes  {label_classes}")
        

        # Compare predictions with ground truth
        tp, fp, fn = compare_labels_vectorized(
            pred_boxes, pred_classes, torch.ones(len(pred_boxes)).to("cuda"),  # dummy conf=1
            label_boxes, label_classes,
            tile_size=640, iou_threshold=0.5, containment_threshold=0.8, convert_to_xyxy=False
        )

        results.append([filename, tp, fp, fn])

        if save_images:
            image_path = os.path.join(base_image_path, filename)
            image = cv2.imread(image_path)
            H, W = image.shape[:2]
            make_image_with_boxes(image, tp, fp, fn, image_output_path, filename)

        metrics = compute_metrics(results)
        if save_results:
            with open(os.path.join(base_output_path, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            save_results_to_json(base_output_path, results)

    end = time.time()
    print(f"Evaluation took {end-start:.2f} seconds.")

evaluate_on_test_set_from_txt()