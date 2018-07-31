import csv


# Class CAD

class CAD:
    '''
    Implementation of the cad model
    '''

    def __init__(self):
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


    def load_cad(self, path='data_files', name_file_faces='cad_faces.csv', name_file_vertices='cad_vertices.csv'):
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
