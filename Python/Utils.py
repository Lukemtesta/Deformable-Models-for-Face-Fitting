
import os

from pprint import pprint


'''
Print all members of an object
'''
def InspectObject(i_obj):

    pprint(vars(i_obj))

'''
Get directory to file

\return full directory to script
'''
def GetDirectory(i_file):

    file_dir = os.path.abspath(i_file)
    file_dir = os.path.dirname(file_dir)

    return file_dir
