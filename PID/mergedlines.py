import cv2
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union
import networkx as nx

def add_pixels_to_lines(lines, num_pixels=50):
    modified_lines = []

    for line in lines:
        modified_line = [(p[0] - num_pixels, p[1] - num_pixels) for p in line]
        modified_lines.append(modified_line)

    return modified_lines

def mergeLines(merged_lines, width, height, buffer=0.05):
    # Pas de lijncoördinaten aan om rond elke lijn een buffer van 5% toe te voegen
    merged_lines = [
        (
            (
                int(start_x - (end_x - start_x) * buffer),
                int(start_y - (end_y - start_y) * buffer),
            ),
            (int(end_x + (end_x - start_x) * buffer), int(end_y + (end_y - start_y) * buffer)),
        )
        for (start_x, start_y), (end_x, end_y) in merged_lines
    ]

    # Trek horizontale lijnen en verticale lijnen recht
    def make_line_straight(line):
        coords = list(line.coords)
        start_x, start_y = coords[0]
        end_x, end_y = coords[-1]
        angle = np.arctan2(end_y - start_y, end_x - start_x)
        # Threshold angle for horizontal or vertical lines
        horizontal_threshold = np.pi / 4  # 45 degrees
        vertical_threshold = np.pi / 2  # 90 degrees

        if abs(angle) < horizontal_threshold:
            return LineString([(start_x, start_y), (end_x, start_y)])
        elif abs(angle - np.pi / 2) < vertical_threshold:
            return LineString([(start_x, start_y), (start_x, end_y)])
        else:
            return line


    # Make all lines in merged_lines straight
    straightened_merged_lines = [
        make_line_straight(LineString(line)) for line in merged_lines
    ]

    # Create LineString objects
    lines = [line for line in straightened_merged_lines if line]

    # Create a graph to represent line intersections
    G = nx.Graph()

    # Add nodes to the graph for each LineString
    for i, line in enumerate(lines):
        G.add_node(i, geometry=line)

    # Check for intersections and add edges to the graph
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if (
                lines[i].intersects(lines[j])
                or lines[i].overlaps(lines[j])
                or lines[i].touches(lines[j])
            ):
                G.add_edge(i, j)

    # Find connected components (clusters) in the graph
    clusters = list(nx.connected_components(G))

    # Find LineStrings that are not part of any cluster
    non_intersecting_lines = [
        lines[i] for i in range(len(lines)) if i not in set.union(*clusters)
    ]
    # print(non_intersecting_lines)

    # Merge LineStrings in each cluster
    merged_lines = []
    for cluster in clusters:
        cluster_lines = [lines[i] for i in cluster]
        merged_cluster = unary_union(cluster_lines)
        merged_lines.append(merged_cluster)

    # Create an image to draw the merged and non-intersecting lines on
    # width, height = wid, 1280  # Adjust the image size as needed
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Define unique colors for drawing the merged lines
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
    ]  # You can add more colors if needed

    # Iterate through the merged_lines and draw LineString objects
    for i, merged_line in enumerate(merged_lines):
        color = colors[i % len(colors)]  # Get a unique color for each line
        if merged_line.geom_type == "MultiLineString":
            for line in merged_line.geoms:
                merged_coords = list(line.coords)
                for j in range(len(merged_coords) - 1):
                    pt1 = tuple(map(int, merged_coords[j]))
                    pt2 = tuple(map(int, merged_coords[j + 1]))
                    cv2.line(image, pt1, pt2, color, 2)
                    cv2.putText(
                        image, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.08, color, 2
                    )
        elif merged_line.geom_type == "LineString":
            merged_coords = list(merged_line.coords)
            for j in range(len(merged_coords) - 1):
                pt1 = tuple(map(int, merged_coords[j]))
                pt2 = tuple(map(int, merged_coords[j + 1]))
                cv2.line(image, pt1, pt2, color, 2)
                cv2.putText(
                    image, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.08, color, 2
                )
        else:
            print(f"Unexpected geometry type: {merged_line.geom_type}")
    
    lines_dict = {}
    for index, value in enumerate(merged_lines):    
        lines_dict[index] = value
    
    cv2.imwrite('test.png', image)
    return lines_dict
  
