import xml.etree.ElementTree as ET
import sys

def extract_XML_obj(filepath):
    objects = []
    try:
        file_text = open(filepath).read()
        file_text = "<root>"+file_text+"</root>"
        children = ET.fromstring(file_text)
        filename = children[0].text
        folder =  children[1].text
        scene = children[2].text

        # for all objects in scene
        for objecto in children[3:]:
            class_num = objecto.find('class').text
            min_x = sys.maxint
            min_y = sys.maxint
            max_x = -1
            max_y = -1

            # for all points outlining object
            for point in objecto.find('polygon').findall('pt'):
                point_y = int(point.find('y').text)
                point_x = int(point.find('x').text)
                min_x = min(point_x, min_x)
                max_x = max(point_x, max_x)
                min_y = min(point_y, min_y)
                max_y = max(point_y, max_y)

            objects.append((class_num, min_x, max_x, min_y, max_y))
    except Exception as e:
        pass

    return objects

