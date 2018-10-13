

class Controller:
    """ Controller is what performs the optimization and tuning of the MZI mesh
        It is initialized with a mesh object and then contains several methods for
        Full optimization and steps.
    """
    def __init__(self, mesh):
        self.mesh = mesh

    def random_perturbation():