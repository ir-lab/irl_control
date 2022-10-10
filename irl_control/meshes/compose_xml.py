from xml.etree.ElementTree import Element, SubElement, tostring
import xml.etree.ElementTree as ET
import os
from xml.dom import minidom

folder = 'basket'
files = [x for x in os.listdir(folder) if x.endswith(".stl")]
root = Element('mujoco')
assets = SubElement(root, "asset")
worldbody = SubElement(root, "worldbody")

for i, file in enumerate(files):
    fname = "".join(file.split('.')[:-1])
    attribs = {
        "name" : fname, 
        "file" : folder + '/' + file
    }
    child = SubElement(assets, "mesh", attrib=attribs)

starting_pos = "0.0 0 0"
attribs = {
    "name" : "basket_whole", 
    "pos" : starting_pos, 
    "quat" : "1.0 0 0 0"
}
sub_piece = SubElement(worldbody, "body", attrib=attribs)

for i, file in enumerate(files):
    fname = "".join(file.split('.')[:-1])
    attribs = {
        "name" : fname, 
        "type" : "mesh",
        "mesh" : fname,
        "pos" : "0 0 0",
        "conaffinity" : "1",
        "contype" : "1",
        # "mass" : "0.001"
    }
    geoms = SubElement(sub_piece, "geom", attrib=attribs)

tree = ET.ElementTree(root)

with open(folder + '/basket.xml', 'wb') as f:
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    f.write(xmlstr.encode('utf-8'))

# with open('out2.xml', 'wb') as f:
#     x = etree.parse("out.xml")
#     f.write(etree.tostring(x, pretty_print=True))
#files = os.listdir('./good_out')

