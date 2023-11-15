# Huy Chu a.k.a. The Ultimate Asian Prodigy 29-09-2023
import os
import datetime
import cv2
import numpy as np
import numpy as np
import matplotlib.image as mpimg
import math
import os
import easyocr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from shapely.geometry import Polygon, LineString, MultiLineString
import shapely.affinity
import pandas as pd
from detectLines import process_lines
from mergedlines import mergeLines

segmentation_class_colors = {
    8: (0, 0, 255),  # Red
    9: (0, 255, 0),  # Green
    10: (255, 0, 0),  # Blue
    11: (255, 255, 0),  # Yellow
    # Add more class-color mappings as needed
}

class ConnectionDetector:
    def __init__(self, image_path, labels_path, labels_segmentation_path):
        self.image_path = image_path
        self.labels_path = labels_path
        self.labels_segmentation_path = labels_segmentation_path
        self.project_dir = ""
        self.binary_path = ""
        self.objectes_removed_path = ""
        self.text_removed_path = ""
        self.unique_lines_path = ""
        self.segmentation_removed_path = ""
        self.object_rectangles = None
        self.text_coordinates = None
        self.merged_lines = None
        self.yolov5_result = None
        self.rectangles_dict = {}
        self.lines_dict = {}
        self.image_width = None
        self.image_height = None
        self.line_colors = {}  # Dictionary to store line colors
        self.segmentation_polygons = {}

    def create_connection_project_dir(self):
        # Define the directory path to check and the folder name
        directory_path = r"connection"
        folder_name = "detect_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Construct the full path to the folder you want to check
        folder_to_check = os.path.join(directory_path, folder_name)
        self.project_dir = folder_to_check
        os.makedirs(folder_to_check)

    def getBinary(self):
        # Load the image
        # Replace 'your_image.jpg' with the path to your image
        image = cv2.imread(self.image_path)
        self.image_height, self.image_width = image.shape[:2]
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # Save or display the binary image
        cv2.imwrite(
            self.project_dir + r"\binary_image.jpg", binary_image
        )  # Save the binary image
  

        self.binary_path = self.project_dir + r"\binary_image.jpg"

        return None

    def removeRectangle(self):
        # Load the image
        # Replace 'your_image.jpg' with the path to your image
        image = cv2.imread(self.text_removed_path)
        rectangles = []
        with open(self.labels_path, "r") as label_file:
            for index, line in enumerate(label_file):
                line = line.split(" ")
                rectangles.append(
                    (float(line[1]), float(line[2]), float(line[3]), float(line[4]))
                )
                self.rectangles_dict[str(index)] = {
                    "rect": [
                        float(line[1]),
                        float(line[2]),
                        float(line[3]),
                        float(line[4]),
                    ],
                    "class": str(line[0]),
                }
        # Create a new dictionary for the updated data
        larger_rect = {}

        # Iterate through the original dictionary and update 'rect' values in the new dictionary
        for key, value in self.rectangles_dict.items():
            rect = value["rect"]
            # Multiply width and height by 1.05 (5% bigger)
            new_rect = [rect[0], rect[1], rect[2] * 1.1, rect[3] * 1.1]
            # Create a new entry in the updated dictionary
            larger_rect[key] = {"rect": new_rect, "class": value["class"]}

        # Iterate through the rectangles and draw white rectangles on the image
        for key, value in larger_rect.items():
            x, y, w, h = value["rect"]
            image_height, image_width, _ = image.shape
            x1 = int((x - w / 2) * image_width)
            y1 = int((y - h / 2) * image_height)
            x2 = int((x + w / 2) * image_width)
            y2 = int((y + h / 2) * image_height)

            # Create a white rectangle with the same dimensions as the region
            white_rectangle = np.ones((y2 - y1, x2 - x1, 3), dtype=np.uint8) * 255

            # Replace the region in the original image with the white rectangle
            image[y1:y2, x1:x2] = white_rectangle

        # Save the modified image
        cv2.imwrite(self.project_dir + r"\yolov5_objects_removed.jpg", image)
        self.objectes_removed_path = self.project_dir + r"\yolov5_objects_removed.jpg"
        self.object_rectangles = rectangles
        return None

    def removeSegments(self):
        # Create an empty dictionary to store the coordinates for each class
        class_coordinates = {}

        # Read the YOLOv8 label text file
        with open(self.labels_segmentation_path, "r") as file:
            lines = file.readlines()

        # Iterate through each line in the file
        for line in lines:
            data = line.strip().split()  # Split the line into tokens
            class_id = int(data[0])  # Extract the class ID

            # Check if there are enough values (at least 5: class ID and 4 coordinates)
            if len(data) >= 5:
                # Extract the coordinates as floats and store them in a tuple
                coordinates = tuple(map(float, data[1:]))
                if class_id not in class_coordinates:
                    class_coordinates[class_id] = []
                class_coordinates[class_id].append(coordinates)

        # Load the image
        image = cv2.imread(self.objectes_removed_path)
        mask = np.zeros_like(image, dtype=np.uint8)

        id = 1
        for class_id, coordinates_list in class_coordinates.items():
            for coordinates in coordinates_list:
                # Split the coordinates into x and y values
                x_values = coordinates[
                    ::2
                ]  # Take every other value starting from index 0
                y_values = coordinates[
                    1::2
                ]  # Take every other value starting from index 1

                # Convert the coordinates to a list of 2D coordinate tuples
                coordinates_2d = [
                    (int(x * image.shape[1]), int(y * image.shape[0]))
                    for x, y in zip(x_values, y_values)
                ]

                # Create a Shapely polygon from the 2D coordinates
                polygon = Polygon(coordinates_2d)

                scale_factor = 1.1

                # Scale the polygon by scaling each coordinate and preserving the centroid
                centroid = polygon.centroid
                scaled_polygon = shapely.affinity.scale(
                    polygon, xfact=scale_factor, yfact=scale_factor, origin=centroid
                )

                # Convert the scaled polygon's coordinates to a NumPy array
                polygon_coords = np.array(
                    list(scaled_polygon.exterior.coords), dtype=np.int32
                )

                # Create a white mask for the current class
                cv2.fillPoly(image, [polygon_coords], (255, 255, 255))

                # Append the scalled polygons to class dictionary
                self.segmentation_polygons[id] = {
                    "polygon": scaled_polygon,
                    "class": class_id,
                }
                id += 1

        print(self.segmentation_polygons)

        cv2.imwrite(self.project_dir + r"\segmentation_removed.jpg", image)
        self.segmentation_removed_path = self.project_dir + r"\segmentation_removed.jpg"

    def detectText(self):
        # Load the image
        image_path = self.binary_path
        image = cv2.imread(image_path)

        # Create an EasyOCR reader for English
        reader = easyocr.Reader(["en"])

        # Perform text recognition
        result = reader.readtext(image_path)

        # Iterate through the detected text regions and draw bounding boxes with text overlay
        for bbox, text, prob in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Fill the bounding box with white color
            cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)

        # Save the modified image
        cv2.imwrite(self.project_dir + r"\yolov5_text_removed.jpg", image)
        self.text_removed_path = self.project_dir + r"\yolov5_text_removed.jpg"

    def drawLines(self):
        # Create a white canvas to draw the lines
        # Adjust the size as needed
        canvas = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        # Generate unique colors for each line
        colors = np.random.randint(
            0, 256, size=(len(self.merged_lines), 3), dtype=np.uint8
        )

        # Draw each line with a unique color
        for i, line in enumerate(self.merged_lines):
            color = tuple(map(int, colors[i]))  # Convert the color to tuple
            cv2.line(canvas, line[0], line[1], color, 2)

        # Save the image with the lines drawn
        cv2.imwrite(self.project_dir + r"\lines_with_unique_colors.jpg", canvas)
        self.unique_lines_path = self.project_dir + r"\lines_with_unique_colors.jpg"

    def drawShapes(self):
        image = cv2.imread(self.unique_lines_path)

        image_height, image_width, _ = image.shape

        canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        # Draw YOLOv5 shapes (rectangles) on the canvas
        for key, value in self.rectangles_dict.items():
            x, y, w, h = value["rect"]
            x1 = int((x - w / 2) * image_width)
            y1 = int((y - h / 2) * image_height)
            x2 = int((x + w / 2) * image_width)
            y2 = int((y + h / 2) * image_height)
            # Draw a green rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw lines on the canvas
        for line in self.merged_lines:
            point1, point2 = line
            x1, y1 = point1
            x2, y2 = point2
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw a red line

        # If you want to save the canvas image
        cv2.imwrite(self.project_dir + r"\shapes.jpg", canvas)

    def get_unique_color(self, key):
        # Generate and return a unique color for the given key
        if key not in self.line_colors:
            # Generate random BGR color values
            color = (
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256),
            )
            self.line_colors[key] = color
        return self.line_colors[key]

    def checkIntersections(self):
        # Increase the width and height by 10 percent
        for key, values in self.rectangles_dict.items():
            rect_values = values["rect"]
            # Multiply the last two values (width and height) by 1.1
            rect_values[2] *= 1.1
            rect_values[3] *= 1.1
            values["rect01"] = rect_values

        lines = self.merged_lines
        lines_dict = mergeLines(lines, self.image_width, self.image_height)

        # Iterate over yolov5_boxes and check for polygon-line intersections
        for key, values in self.rectangles_dict.items():
            rect01_values = values["rect01"]

            # Extract the four values (x, y, width, height) from 'rect01'
            x, y, width, height = rect01_values

            # Calculate the coordinates of the four corners of the polygon
            x_min = x - width / 2
            y_min = y - height / 2
            x_max = x + width / 2
            y_max = y + height / 2

            # Normalize to a 1280x1280 canvas
            x_min *= self.image_width
            y_min *= self.image_height
            x_max *= self.image_width
            y_max *= self.image_height

            # Create a Polygon object using Shapely
            polygon = Polygon(
                [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            )

            # Add the polygon to the dictionary under the key 'polygon'
            values["polygon"] = polygon

        # Iterate over yolov5_boxes and check for polygon-line intersections
        for key1, box1 in self.rectangles_dict.items():
            polygon = box1["polygon"]
            line_connections = []

            for key2, line_info in lines_dict.items():
                line_string = line_info

                if (
                    polygon.intersects(line_string)
                    or polygon.touches(line_string)
                    or polygon.overlaps(line_string)
                ):
                    line_connections.append(key2)

            if line_connections:
                box1["lineconnection"] = line_connections
            else:
                box1["lineconnection"] = []

        # Iterate over segmentation polygons and check for polygon-line intersections
        for key3, box3 in self.segmentation_polygons.items():
            polygon = box3["polygon"]
            line_connections = []

            for key4, line_info in lines_dict.items():
                line_string = line_info

                if (
                    polygon.intersects(line_string)
                    or polygon.touches(line_string)
                    or polygon.overlaps(line_string)
                ):
                    line_connections.append(key4)

            if line_connections:
                box3["lineconnection"] = line_connections
            else:
                box3["lineconnection"] = []

        # Create a figure and axis with a 1280x1280 canvas
        fig, ax = plt.subplots(
            figsize=(self.image_width / 100, self.image_height / 100), dpi=100
        )

        # Plot polygons and label them with unique IDs
        for key, values in self.rectangles_dict.items():
            polygon = values.get("polygon")  # Check if 'polygon' key exists
            if polygon:
                x, y = polygon.exterior.xy
                ax.fill(x, y, alpha=0.5, label=f"Polygon {key}")

        # Plot line strings and label them with unique IDs from self.lines_dict
        for key, values in lines_dict.items():
            line_geometry = values  # Check if 'large' key exists
            if line_geometry:
                if isinstance(line_geometry, MultiLineString):
                    # Handle MultiLineStrings
                    for i, line in enumerate(line_geometry.geoms):
                        x, y = zip(*line.xy)
                        ax.plot(x, y, label=f"MultiLineString {key}-{i}", linewidth=2)
                        ax.text(
                            x[0],
                            y[0],
                            f"MultiLineString {key}-{i}",
                            fontsize=10,
                            color="r",
                            backgroundcolor="w",
                        )
                else:
                    # Handle LineStrings
                    x, y = zip(*line_geometry.xy)
                    ax.plot(x, y, label=f"LineString {key}", linewidth=2)
                    ax.text(
                        x[0],
                        y[0],
                        f"LineString {key}",
                        fontsize=10,
                        color="r",
                        backgroundcolor="w",
                    )

        # Add legends to distinguish polygons and line strings
        ax.legend()

        # Set axis limits based on your canvas size (1280x1280)
        ax.set_xlim(0, 1280)  # Adjust as needed
        ax.set_ylim(0, 1280)  # Adjust as needed

        # Invert the y-axis if necessary
        plt.gca().invert_yaxis()

        # Show the plot
        plt.gca().set_aspect("equal", adjustable="box")  # Maintain aspect ratio

        # Display the figure with unique identifiers
        for key, values in self.rectangles_dict.items():
            polygon = values.get("polygon")  # Check if 'polygon' key exists
            if polygon:
                x, y = polygon.exterior.xy
                ax.fill(x, y, alpha=0.5, label=f"Polygon {key}")
                ax.text(
                    x[0],
                    y[0],
                    f"Polygon {key}",
                    fontsize=10,
                    color="r",
                    backgroundcolor="w",
                )

        # Plot line strings and label them with unique IDs from self.lines_dict
        for key, values in lines_dict.items():
            line_geometry = values  # Check if 'large' key exists
            if line_geometry:
                if isinstance(line_geometry, MultiLineString):
                    # Handle MultiLineStrings
                    for i, line in enumerate(line_geometry.geoms):
                        x, y = zip(*line.xy)
                        ax.plot(x, y, label=f"MultiLineString {key}-{i}", linewidth=2)
                        ax.text(
                            x[0],
                            y[0],
                            f"MultiLineString {key}-{i}",
                            fontsize=10,
                            color="r",
                            backgroundcolor="w",
                        )
                else:
                    # Handle LineStrings
                    x, y = zip(*line_geometry.xy)
                    ax.plot(x, y, label=f"LineString {key}", linewidth=2)
                    ax.text(
                        x[0],
                        y[0],
                        f"LineString {key}",
                        fontsize=10,
                        color="r",
                        backgroundcolor="w",
                    )

        # plt.show()
        plt.savefig(
            self.project_dir + r"\intersections.png", dpi=100, bbox_inches="tight"
        )

        # Create an empty canvas with a white background
        canvas = np.ones((self.image_height, self.image_width, 3), np.uint8) * 255

        # Plot polygons and label them with unique IDs
        for key, values in self.rectangles_dict.items():
            polygon = values.get("polygon")  # Check if 'polygon' key exists
            if polygon:
                pts = np.array(polygon.exterior.coords, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(canvas, [pts], (0, 0, 255))  # Fill polygon with red
                cv2.polylines(
                    canvas, [pts], isClosed=True, color=(0, 0, 0), thickness=2
                )  # Draw polygon boundary

                # Add text label
                x, y = polygon.exterior.coords[0]
                cv2.putText(
                    canvas,
                    f"Polygon {key}",
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

        # Plot polygons and label them with unique IDs
        for key, values in self.segmentation_polygons.items():
            polygon = values.get("polygon")
            class_id = values.get("class")
            if polygon and class_id in segmentation_class_colors:
                color = segmentation_class_colors[class_id]

                pts = np.array(polygon.exterior.coords, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(
                    canvas, [pts], color
                )  # Fill polygon with class-based color
                cv2.polylines(
                    canvas, [pts], isClosed=True, color=(0, 0, 0), thickness=2
                )  # Draw polygon boundary

                # Add text label
                x, y = polygon.exterior.coords[0]
                cv2.putText(
                    canvas,
                    f"Polygon {key} (Class {class_id})",
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

        # Plot line strings and label them with unique IDs from self.lines_dict
        for key, values in lines_dict.items():
            line_geometry = values  # Check if 'large' key exists
            if line_geometry:
                if isinstance(line_geometry, MultiLineString):
                    # Handle MultiLineStrings
                    color = self.get_unique_color(f"MultiLineString {key}")
                    for i, line in enumerate(line_geometry.geoms):
                        pts = np.array(line.coords, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(
                            canvas, [pts], isClosed=False, color=color, thickness=2
                        )  # Draw line with unique color

                        # Add text label
                        x, y = line.coords[0]
                        cv2.putText(
                            canvas,
                            f"MultiLineString {key}-{i}",
                            (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                        )
                else:
                    # Handle LineStrings
                    pts = np.array(line_geometry.coords, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    color = self.get_unique_color(f"LineString {key}")
                    cv2.polylines(
                        canvas, [pts], isClosed=False, color=color, thickness=2
                    )  # Draw line with unique color

                    # Add text label
                    x, y = line_geometry.coords[0]
                    cv2.putText(
                        canvas,
                        f"LineString {key}",
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

        # Save the image
        cv2.imwrite(self.project_dir + r"\intersections.png", canvas)

    def detectTextShapes(self):
        yolo_results = self.object_rectangles

        # Load the image
        image_path = self.binary_path  # Replace with the path to your image
        image = cv2.imread(image_path)

        # Initialize EasyOCR reader
        # You can specify the languages you want to support
        reader = easyocr.Reader(["en"])

        # Create an empty list to store the detected text
        detected_text_list = []

        # Loop over the YOLOv5 results
        for index, value in self.rectangles_dict.items():
            x, y, w, h = value["rect01"]
            # Convert YOLO coordinates to pixel coordinates
            img_height, img_width, _ = image.shape
            left = int((x - w / 2) * img_width)
            top = int((y - h / 2) * img_height)
            right = int((x + w / 2) * img_width)
            bottom = int((y + h / 2) * img_height)

            # Ensure valid coordinates
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            if right > img_width:
                right = img_width
            if bottom > img_height:
                bottom = img_height

            # Crop the bounding box region from the image
            bbox_image = image[top:bottom, left:right]

            # Check if the cropped image is empty
            if bbox_image is not None and bbox_image.size > 0:
                # Perform text detection using EasyOCR on the cropped region
                results = reader.readtext(bbox_image)

                detected_texts = []
                for detected_result in results:
                    detected_text = detected_result[1]
                    detected_texts.append(detected_text)

                # Save the detected text to self.rectangles_dict under the key 'foundtext'
                self.rectangles_dict[index]["detectedtext"] = detected_texts
            else:
                # Handle the case where the cropped image is empty
                print(f"Warning: Cropped image is empty for index {index}")

    def exportToExcel(self):
        # Create a list of dictionaries to prepare the data for the DataFrame
        data_list = [
            {
                "ID": key,
                "Class": value["class"],
                "Rect": value["rect"],
                "detectedtext": ", ".join(map(str, value["detectedtext"])),
                "lineconnection": ", ".join(map(str, value["lineconnection"])),
            }
            for key, value in self.rectangles_dict.items()
        ]
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data_list)

        # Display the DataFrame
        df.to_excel(
            self.project_dir + r"\connections.xlsx",
            index=False,
            sheet_name="RectangleConnections",
        )

        # Create a DataFrame
        df_segmentation = pd.DataFrame(
            self.segmentation_polygons.values(), index=self.segmentation_polygons.keys()
        )

        # Rename the columns
        df_segmentation.columns = ["Polygon", "Class", "Lineconnection"]

        # Add the ID column
        df_segmentation["ID"] = df_segmentation.index

        # Reorder the columns
        df_segmentation = df_segmentation[["ID", "Class", "Lineconnection"]]

        # Define the Excel file name
        excel_file = self.project_dir + r"\connections.xlsx"

        # Define the sheet name where you want to write the data
        sheet_name = "SegmentationConnections"  # Change this to the desired sheet name

        # Write the DataFrame to the Excel file
        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a") as writer:
            df_segmentation.to_excel(writer, sheet_name=sheet_name, index=False)

    def runDetection(self):
        self.create_connection_project_dir()
        self.getBinary()
        self.detectText()
        self.removeRectangle()
        self.removeSegments()
        self.merged_lines = process_lines(
            self.segmentation_removed_path, self.project_dir + r"\lines_detected.jpg"
        )
        self.drawLines()
        self.drawShapes()
        self.checkIntersections()
        self.detectTextShapes()
        self.exportToExcel()
        print(self.segmentation_polygons)
        os.startfile(self.project_dir)


path = r"D:\Projects\School\MateriaalEngineeringDiagrams\blok3_22-23\Automated_Diagrams_Code\yolov5_ws\detect\p_id_6_0.png"
labels = r"D:\Projects\School\MateriaalEngineeringDiagrams\blok3_22-23\Automated_Diagrams_Code\yolov5_ws\yolov5\runs\detect\exp61\labels\p_id_6_0.txt"
labels_segmentation = (
    r"C:\Users\huy_c\Desktop\Projects\yolov5\runs\segment\predict11\labels\p_id_6_0.txt"
)
detection = ConnectionDetector(path, labels, labels_segmentation)

# test.checkIntersections()
detection.runDetection()
