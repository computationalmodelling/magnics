import fenics as fe

def vectorspace(mesh):
    """setup"""
    V = fe.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    v = fe.TestFunction(V)
    u = fe.TrialFunction(V)
    m = fe.Function(V)
    Heff = fe.Function(V)
    return m, Heff, u, v, V
