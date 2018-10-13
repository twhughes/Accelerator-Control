

class ControllerFactory:
    """ Controller is what performs the optimization and tuning of the MZI mesh
        It is initialized with a mesh object and then contains several methods for
        Full optimization and steps.
    """
    def __init__(self, mesh):
        self.mesh_type = mesh.mesh_type

    def create(self, params):
        if self.mesh_type == 'triangular':
            controller = ControllerTriangle(params)
        else:
            controller = ControllerClements(params)
        return controller


class ControllerTriangle:

    def __init__(self, params):
        self.params = params
        self.poo = 'ppoo'

    def triangle_method(self):
        pass

class ControllerClements:

    def __init__(self, params):
        self.params = params
        self.poo = 'caca'    

    def clements_method(self):
        pass