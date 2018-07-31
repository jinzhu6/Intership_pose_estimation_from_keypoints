import csv
import numpy as np


# Class CAD

class Model:
    '''
    Implementation of the cad model
    '''

    def __init__(self):
        self.nb_faces = 0
        self.nb_vertices = 0
        self.faces = []
        self.vertices = []
        self.pname = ['fru', 'frd', 'flu', 'fld', 'bru', 'brd', 'blu', 'bld']
        self.fru = [0.0800, 0.1150, 0.0300]
        self.frd = [0.0800, 0.1150, -0.0300]
        self.flu = [0.0800, -0.1150, 0.0300]
        self.fld = [0.0800, -0.1150, -0.0300]
        self.bru = [-0.0800, 0.1150, 0.0300]
        self.brd = [-0.0800, 0.1150, -0.0300]
        self.blu = [-0.0800, -0.1150, 0.0300]
        self.bld = [-0.0800, -0.1150, -0.0300]
        self.scale = [0.16, 0.23, 0.06]
        self.kp = [
            [self.scale[0] / 2, self.scale[1] / 2, self.scale[2] / 2],
            [self.scale[0] / 2, self.scale[1] / 2, -self.scale[2] / 2],
            [self.scale[0] / 2, -self.scale[1] / 2, self.scale[2] / 2],
            [self.scale[0] / 2, -self.scale[1] / 2, -self.scale[2] / 2],
            [-self.scale[0] / 2, self.scale[1] / 2, self.scale[2] / 2],
            [-self.scale[0] / 2, self.scale[1] / 2, -self.scale[2] / 2],
            [-self.scale[0] / 2, -self.scale[1] / 2, self.scale[2] / 2],
            [-self.scale[0] / 2, -self.scale[1] / 2, -self.scale[2] / 2]]


    def load_model(self, path='data_files', name_file_faces='cad_faces.csv', name_file_vertices='cad_vertices.csv'):
        # loads the cad model into self.faces and self.vertices

        # faces loading
        self.faces = []
        name_faces = path + '/' + name_file_faces

        file_faces = open(name_faces, 'r')
        count = 0

        for lines in file_faces:
                line = lines.split(',')
                self.faces.append([])

                for x in line:
                    self.faces[count].append(float(x))

                count += 1

        file_faces.close()

        #vertices loading
        self.vertices = []
        name_vertices = path + '/' + name_file_vertices

        files_vertices = open(name_vertices, 'r')
        count = 0

        for lines in files_vertices:
                line = lines.split(',')
                self.vertices.append([])

                for x in line:
                    self.vertices[count].append(float(x))

                count += 1

        files_vertices.close()

        self.nb_faces = len(self.faces)
        self.nb_vertices = len(self.vertices)




class Template:
    '''
    Implementation of the dict object of the matlab code
    '''

    def __init__(self, model):
        self.nbjoint = 8 # nb of joints
        self.m = 3 # dimension, if it is not 3 i really want to know what are you fucking doing
        self.kp_id = [i for i in range(self.nbjoint)] # id of the joints
        self.kp_name = ["fru","frd","flu","fld","bru","brd","blu","bld"] # name of the joints
        self.pc = [] # I have no idea what it is

        # S is the scale matrix of the model, mu = S
        S = np.transpose(model.kp)
        self.mu = S

        mean = np.mean(S,1)
        for i in range(self.m):
            S[i] -= mean[i]

        std = np.std(S,1)
        a = np.mean(std)
        S = S / a

        self.B = S # B is the normalized version of S, this normalisation is pretty stange

