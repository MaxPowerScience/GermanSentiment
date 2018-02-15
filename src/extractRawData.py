"""Author:Manuel Reinbold, Maximilian Renk
Date: 21/11/17
Version: 1.0
"""

from xml.dom import minidom

def main():
    # parse an xml file by name
    mydoc = minidom.parse('C:/Users/Max/Downloads/train_v1.4.xml')
    mydoc = minidom.parse('C:/Users/Max/Downloads/dev_v1.4.xml')

    items = mydoc.getElementsByTagName('Document')

    # one specific item attribute
    print('Item #2 attribute:')
    d = items[0].childNodes
sdfsdfsdf
    # all item attributes
    print(len(items))

    print('\nAll attributes:')
    for elem in items:
        for child in elem.childNodes:
            if child.nodeName == 'sentiment':
                print(child.firstChild.data)
                a = 3
            if child.nodeName == 'text':
                print(child.firstChild.data)


if __name__ == "__main__":
    main()