import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from PEPit import PEP
import PEPit
from PEPit.functions import SmoothStronglyConvexFunction, SmoothConvexFunction, SmoothFunction
import cvxpy as cp
from PEPit.primitive_steps import inexact_gradient_step
import gurobipy as gp
from gurobipy import GRB
from cvxpy import Variable, Problem, Minimize, Maximize, SCS, MOSEK, PSD

def wc_gradient_descent(mu, L, gamma, n, verbose=1, metric = 0, dual_multipliers = False, worst_case_function=False):
    """
    Args:
        mu (float): the strongly-convex constant
        L (float): the smoothness parameter.
        gamma (float or list): step-size (either constant or non-constant)
        n (int): number of iterations.
        verbose (int): Level of information details to print.
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.
        metric (int): Metric used to evaluate the convergence.
                        - 0: ||x_n - x_*||^2 <= c_0 ||x_0 - x_*||^2
                        - 1: f(x_n) - f_* <= c_1 ||x_0 - x_*||^2
                        - 2: nabla f(x_n)^2 <= c_2 (f(x_0) - f_*)
                        - 3: nabla f(x_n)^2 <= c_3 ||x_0 - x_*||^2 
    """
    
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    if mu <= -1:
        func = problem.declare_function(function_class=SmoothFunction, L = L)
    elif mu == 0:
        func = problem.declare_function(SmoothConvexFunction, L = L)
    else:
        func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Run n steps of the GD method
    x = x0
    try: # If gamma is a constant (float) stepsize
        for i in range(n):
            x = x - gamma * func.gradient(x)
    except: # If gamma is a non-constant (list) stepsize
        for i in range(n):
            x = x - gamma[i] * func.gradient(x)
            # print([i.eval_dual() for i in problem._list_of_constraints_sent_to_cvxpy])
            
    # Set the initial constraint that is the distance between x0 and x^*
    if metric == 0 or metric == 1 or metric == 3:
        problem.set_initial_condition((x0 - xs) ** 2 <= 1) #problem.set_initial_condition(func(x0) - fs <= 1) #
    elif metric == 2:
        problem.set_initial_condition((func(x0) - func(xs)) <= 1)   #Change here: metrix becomes grad <= c (f(x0) - f(xN))
        
    # Set the performance metric to the function values accuracy
    if metric == 0:
        problem.set_performance_metric((x - xs) ** 2)
    elif metric == 1:
        problem.set_performance_metric(func(x) - fs)
    elif metric == 2 or metric == 3:
        problem.set_performance_metric(func.gradient(x) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper = "cvxpy", solver = "MOSEK", tol_dimension_reduction = 1e-6, dimension_reduction_heuristic='trace', verbose=pepit_verbose)
    if dual_multipliers:
        print(func.get_class_constraints_duals())
        print(PEPit.Wrapper.get_primal_variables())

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * (2 * n * L * gamma + 1)) # To change

    # Print conclusion if required
    if verbose != -1:
        if metric == 0:
            print('\tPEPit guarantee:\t ||x_n - x_*||^2 <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        elif metric == 1:
            print('\tPEPit guarantee:\t f(x_n) - f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        elif metric == 2:
            print('\tPEPit guarantee:\t nabla f(x_n)^2 <= {:.6} (f(x_0) - f_*)'.format(pepit_tau))
        elif metric == 3:
            print('\tPEPit guarantee:\t nabla f(x_n)^2 <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))

    return pepit_tau, theoretical_tau

def inexact_wc_gradient_descent(mu, L, gamma, n, verbose=1, metric = 0, epsilon = 0., dual_multipliers = False, worst_case_function=False):
    """
    Args:
        mu (float): the strongly-convex constant
        L (float): the smoothness parameter.
        gamma (float or list): step-size (either constant or non-constant)
        n (int): number of iterations.
        verbose (int): Level of information details to print.
                        - -1: No verbose at all.
                        - 0: This example's output.
                        - 1: This example's output + PEPit information.
                        - 2: This example's output + PEPit information + CVXPY details.
        metric (int): Metric used to evaluate the convergence.
                        - 0: ||x_n - x_*||^2 <= c_0 ||x_0 - x_*||^2
                        - 1: f(x_n) - f_* <= c_1 ||x_0 - x_*||^2
                        - 2: nabla f(x_n)^2 <= c_2 (f(x_0) - f_*)
                        - 3: nabla f(x_n)^2 <= c_3 ||x_0 - x_*||^2
        epsilon (float) : level of inaccuracy
    """
    
    # Instantiate PEP
    problem = PEP()

    # Declare a strongly convex smooth function
    if mu <= -1:
        func = problem.declare_function(SmoothFunction, L = L)
    elif mu == 0:
        func = problem.declare_function(SmoothConvexFunction, L = L)
    else:
        func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Run n steps of the GD method
    x = x0
    try: # If gamma is a constant (float) stepsize
        for i in range(n):
            x, dx, fx = inexact_gradient_step(x, func, gamma=gamma, epsilon=epsilon, notion='relative')
    except: # If gamma is a non-constant (list) stepsize
        for i in range(n):
            x, dx, fx = inexact_gradient_step(x, func, gamma=gamma[i], epsilon=epsilon, notion='relative')
            
    # Set the initial constraint that is the distance between x0 and x^*
    if metric == 0 or metric == 1 or metric == 3:
        problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    elif metric == 2:
        problem.set_initial_condition((func(x0) - func(xs)) <= 1)

    # Set the performance metric to the function values accuracy
    if metric == 0:
        problem.set_performance_metric((x - xs) ** 2)
    elif metric == 1:
        problem.set_performance_metric(func(x) - fs)
    elif metric == 2 or metric == 3:
        problem.set_performance_metric(func.gradient(x) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose, solver = cp.MOSEK, dimension_reduction_heuristic="trace", tol_dimension_reduction=1e-5)
    if dual_multipliers:
        print(func.get_class_constraints_duals())

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * (2 * n * L * gamma + 1)) # To change

    # Print conclusion if required
    if verbose != -1:
        if metric == 0:
            print('\tPEPit guarantee:\t ||x_n - x_*||^2 <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        elif metric == 1:
            print('\tPEPit guarantee:\t f(x_n) - f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        elif metric == 2:
            print('\tPEPit guarantee:\t nabla f(x_n)^2 <= {:.6} (f(x_n) - f_*)'.format(pepit_tau))
        elif metric == 3:
            print('\tPEPit guarantee:\t nabla f(x_n)^2 <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau

def plot_scatter(x, y, color = 'blue', label = '', prefixe = plt):
    prefixe.plot(x, y, color, label = label)
    prefixe.scatter(x, y, c = color)


def PV(h, delta=0, print_=False, n_steps = 1, simpler_multipliers = False):
    # Setup formulation
    L01 = Variable()
    L10 = Variable()
    C = Variable()
    if n_steps == 1:
        cons = [L01 >= 0, L10 >= 0,-L10 + L01 == 1] # ==1 if g1 - C * g0, ==0 else (for the function values to go away)

        if delta == 0:
            A01 = np.array([[1/2, -1/2 + h/2], [-1/2 + h/2, 1/2]])
            A10 = np.array([[1/2 - h, -1/2], [-1/2, 1/2]])
            g1  = np.array([[0, 0], [0, 1]]) #g1
            g0  = np.array([[1, 0], [0, 0]]) #g0
            cons.append(L10 * A10 + L01 * A01 >> C * g1) #g1^2 <= tau_1 (f0-f1)
        else:
            LAP = Variable()
            cons.append(LAP >= 0)
            
            A01 = np.array([[1/2, -1/2 + h/2, 0], [-1/2 + h/2, 1/2, h/2], [0, h/2, 0]])
            A10 = np.array([[1/2 - h, -1/2, -h/2], [-1/2, 1/2, 0], [-h/2, 0, 0]])
            g1  = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) #g1
            g0  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) #g0   
            
            AAP = np.array([[-delta**2, 0, 0], [0, 0, 0], [0, 0, 1]])

            if simpler_multipliers:
                denominator = 1/(h*(1+delta)-1)
                cons = [L01 >= 0, L10 >= 0]
            cons.append(L10 * A10 + L01 * A01 + LAP * AAP >> C * g1)
    if n_steps == 2:
        L02 = Variable()
        L20 = Variable()
        L12 = Variable()
        L21 = Variable()
        cons = [L10 >= 0, L01 >= 0, L02 >=0, L20 >= 0, L12 >= 0, L21 >= 0, L01 + L02 - L10 - L20 == 1, L12 + L10 - L21 - L01 == 0, L20 + L21 - L02 - L12 == -1] #L01 + L02 - L10 - L20 == 0, L12 + L10 - L21 - L01 == 0, L20 + L21 - L02 - L12 == 0 if g1 - C * g0 
        cons.append(L02 == 0)
        cons.append(L20 == 0)

        if delta == 0:
            A01 = np.array([[1/2, -1/2 + h/2, 0], [-1/2 + h/2, 1/2, 0], [0, 0, 0]])
            A10 = np.array([[1/2 - h, -1/2, 0], [-1/2, 1/2, 0], [0, 0, 0]])
            A02 = np.array([[1/2, 0, -1/2 + h/2], [0, 0, h/2], [-1/2 + h/2, h/2, 1/2]])
            A20 = np.array([[1/2 - h, -h/2, -1/2], [-h/2, 0, 0], [-1/2, 0, 1/2]])
            A12 = np.array([[0, 0, 0], [0, 1/2, -1/2 + h/2], [0, -1/2 + h/2, 1/2]])
            A21 = np.array([[0, 0, 0], [0, 1/2 - h, -1/2], [0, -1/2, 1/2]])

            g2  = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
            g1  = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) #g1
            g0  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) #g0
            

            cons.append(L10 * A10 + L01 * A01 + L20 * A20 + L02 * A02 + L12 * A12 + L21 * A21 >> C * g2)
        else: #Order or variables: g0, g1, g2, d0, d1
            LAP_0 = Variable()
            cons.append(LAP_0 >= 0)

            LAP_1 = Variable()
            cons.append(LAP_1 >= 0)
            
            A01 = np.array([[1/2, -1/2 + h/2, 0, 0, 0], [-1/2 + h/2, 1/2, 0, h/2, 0], [0, 0, 0, 0, 0], [0, h/2, 0, 0, 0], [0, 0, 0, 0, 0]])
            A10 = np.array([[1/2 - h, -1/2, 0, -h/2, 0], [-1/2, 1/2, 0, 0, 0], [0, 0, 0, 0, 0], [-h/2, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
            A02 = np.array([[1/2, 0, -1/2 + h/2, 0, 0], [0, 0, h/2, 0, 0], [-1/2 + h/2, h/2, 1/2, h/2, h/2], [0, 0, h/2, 0, 0], [0, 0, h/2, 0, 0]])
            A20 = np.array([[1/2 - h, -h/2, -1/2, -h/2, -h/2], [-h/2, 0, 0, 0, 0], [-1/2, 0, 1/2, 0, 0], [-h/2, 0, 0, 0, 0], [-h/2, 0, 0, 0, 0]])
            A12 = np.array([[0, 0, 0, 0, 0], [0, 1/2, -1/2 + h/2, 0, 0], [0, -1/2 + h/2, 1/2, 0, h/2], [0, 0, 0, 0, 0], [0, 0, h/2, 0, 0]])
            A21 = np.array([[0, 0, 0, 0, 0], [0, 1/2 - h, -1/2, 0, -h/2], [0, -1/2, 1/2, 0, 0], [0, 0, 0, 0, 0], [0, -h/2, 0, 0, 0]])
            AAP_0 = np.array([[-delta**2, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
            AAP_1 = np.array([[0, 0, 0, 0, 0], [0, -delta**2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

            g2  = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) #g2
            g1  = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) #g1
            g0  = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) #g0
            cons.append(L10 * A10 + L01 * A01 + L20 * A20 + L02 * A02 + L12 * A12 + L21 * A21 + LAP_0 * AAP_0 + LAP_1 * AAP_1 >> C * g2) #g1^2 <= tau_1 (f0-f1)

    # Solve
    prob = Problem(Maximize(C), cons)  #min(C) for  g1 - C * g0, min(-C) else
    prob.solve(solver='MOSEK')

    if prob.status == "optimal":
        C = C.value.item()
    else:
        C = np.nan
    
    if n_steps == 1:
        if delta == 0:
            L10_val = L10.value.item()
            L01_val = L01.value.item()
            LAP_val = None
        else:
            L10_val = L10.value.item()
            L01_val = L01.value.item()
            LAP_val = LAP.value.item()
        if print_:
            if delta ==0:
                print("One step with h={} and delta={} -> C={}".format(np.round(h, 2), delta, np.round(C,2)))
                print("Multipliers:    lambda_01={}, lambda_10={}".format(np.round(L01_val,2), np.round(L10_val,2)))
            else:
                print("One step with h={} and delta={} -> C={}".format(np.round(h,2), delta, np.round(C,2)))
                print("Multipliers:    lambda_01={}, lambda_10={}, lambda_apx={}".format(np.round(L01_val,2), np.round(L10_val,2), np.round(LAP_val,3)))
    elif n_steps == 2:
        return C, L01.value.item(), L10.value.item(), L02.value.item(), L20.value.item(), L12.value.item(), L21.value.item(), LAP_0.value.item(), LAP_1.value.item()  
        
    return C, L01.value.item(), L10.value.item(), LAP_val


def non_convex_pep_like(h, delta, verbose = False, truncated_mantissa = False, intermediate = False, rounding_nearest = False, two_d = False): 

    # Create model
    m=gp.Model("PEP_like")
    m.setParam("OutputFlag", 0)      # See logs to understand timing
    m.setParam("NonConvex", 2)       # Required for non-convex QP
    m.setParam("Threads", 1)         # Reduce noise from multi-threading
    m.setParam("Presolve", 2)        # Try aggressive presolve
    m.setParam("TimeLimit", 1)      # Hard cutoff for test runs
    m.setParam("MIPGap", 1e-8)       # Loosen optimality gap (if allowed)

    # Create 2D vector variables
    x0 = m.addVars(2, lb=-50, ub=50, name="x0")
    x1 = m.addVars(2, lb=-50, ub=50, name="x1")
    g0 = m.addVars(2, lb=0, ub=50, name="g0")  #change: lb was 0 here
    g1 = m.addVars(2, lb=-50, ub=50, name="g1")
    d0 = m.addVars(2, lb=0, ub=50, name="d0")  #change: lb was 0 here

    if np.abs(delta) < 1e-6:
        for i in range(2):
            m.addConstr(d0[i] == g0[i])
    # Scalar variables
    f0 = m.addVar(lb=-10, ub=10, name="f0")
    f1 = m.addVar(lb=-10, ub=10, name="f1")

    # Constraint: x1 = x0 - h * d0
    for i in range(2):
        m.addConstr(x1[i] == x0[i] - h * d0[i], name=f"x1_def_{i}")

    # g0[1] = 0, g1[1] = 0  (second component is 0)
    m.addConstr(g0[1] == 0, name="g0_second_zero")
    if two_d:
        m.addConstr(d0[0] == (1-delta**2)*g0[0], name="d0_first_zero")
        m.addConstr(d0[1] == delta * np.sqrt(1 - delta**2) * g0[0], name="d0_second_zero")
    #m.addConstr(g1[0] == g0[0], name="g1_second_zero")

    m.addConstr(x0[0] == 0, name="x0_first_zero")
    m.addConstr(x0[1] == 0, name="x0_second_zero")

    m.addConstr(f0 == 0, name="f_centering")

    # Norm constraints: ||g0 - d0||^2 <= delta^2 * ||g0||^2
    # Define expressions for norms
    g0_minus_d0_sq = m.addVar(lb=0, name="norm_g0_minus_d0_sq")
    g0_sq = m.addVar(lb=0, name="norm_g0_sq")

    if truncated_mantissa:
        m.addConstr(d0[0] <= g0[0], name="d0_first_bound")
        m.addConstr(d0[1] <= g0[1], name="d0_second_bound")
        m.addConstr(0.5*g0[0] <= d0[0])
        m.addConstr(0.5*g0[1] <= d0[1])
    
    if rounding_nearest:
        m.addConstr(d0[0] <= 2*g0[0], name="d0_first_bound")
        m.addConstr(d0[1] <= 2*g0[1], name="d0_second_bound")
        m.addConstr(0.5*g0[0] <= d0[0])
        m.addConstr(0.5*g0[1] <= d0[1])

    m.addQConstr(
        g0_minus_d0_sq == (g0[0] - d0[0]) * (g0[0] - d0[0]) + (g0[1] - d0[1]) * (g0[1] - d0[1]),
        name="norm_g0_minus_d0_sq_def"
    )
    m.addQConstr(
        g0_sq == g0[0] * g0[0] + g0[1] * g0[1],
        name="norm_g0_sq_def"
    )
    m.addConstr(g0_minus_d0_sq <= delta * delta * g0_sq, name="relative_error_constraint")

    # f0 - f1 <= 1
    m.addConstr(f0 - f1 <= 1, name="f_diff_bound")

    # Interpolation constraints
    # (f0 - f1) >= dot(g1, x0 - x1) + 0.5 * ||g0 - g1||^2
    diff1 = m.addVar(lb=0, name="norm_g0_minus_g1_sq_1")
    m.addQConstr(
        diff1 == (g0[0] - g1[0])**2 + (g0[1] - g1[1])**2,
        name="norm_diff1_def"
    )
    dot1 = (g1[0] * (x0[0] - x1[0]) + g1[1] * (x0[1] - x1[1]))
    m.addConstr(
        f0 - f1 >= dot1 + 0.5 * diff1,
        name="interp_f0_f1"
    )

    # (f1 - f0) >= dot(g0, x1 - x0) + 0.5 * ||g1 - g0||^2
    diff2 = m.addVar(lb=0, name="norm_g1_minus_g0_sq_2")
    m.addQConstr(
        diff2 == (g1[0] - g0[0])**2 + (g1[1] - g0[1])**2,
        name="norm_diff2_def"
    )
    dot2 = (g0[0] * (x1[0] - x0[0]) + g0[1] * (x1[1] - x0[1]))
    m.addConstr(
        f1 - f0 >= dot2 + 0.5 * diff2,
        name="interp_f1_f0"
    )
    
    if intermediate:
        m.addConstr(d0[0] == (1-delta**2)*g0[0], name="equality d0 g1")
        m.addConstr(d0[1] == delta*np.sqrt(1-delta**2)*g0[0], name="equality d0 g1")
        m.addConstr(g1[1] == -np.sqrt(1-delta**2)*g1[0])

    # Objective: maximize ||g1||^2
    g1_norm_sq = g1[0] * g1[0] + g1[1] * g1[1]
    m.setObjective(g1_norm_sq, GRB.MAXIMIZE)

    # Solve
    m.optimize()

    # Output
    if verbose:
        if m.status == GRB.OPTIMAL:
            print(f"Optimal ||g1||²: {m.ObjVal}")
            print(f"g0 = {[g0[i].X for i in range(2)]}")
            print(f"g1 = {[g1[i].X for i in range(2)]}")
            print(f"x0 = {[x0[i].X for i in range(2)]}")
            print(f"x1 = {[x1[i].X for i in range(2)]}")
            print(f"d0 = {[d0[i].X for i in range(2)]}")
            print(f"f0 = {f0.X}, f1 = {f1.X}")
        else:
            print("No optimal solution found for h =", h)
    return m.ObjVal, [g0[i].X for i in range(2)], [d0[i].X for i in range(2)], [g1[i].X for i in range(2)]

def my_PEP(h, delta=0, print_=False, n_steps=1, distance_max=np.inf, metric = "min", truncated_mantissa = False):
    try:
        h[0]
        if len(h) != n_steps:
            print("n_steps is different from the size of list of steps")
    except:
        h = h*np.ones(n_steps)

    rate_vector = Variable(n_steps + 1, nonneg=True)
    multipliers_vector = Variable((n_steps + 1, n_steps + 1), nonneg=True)
    inexact_multipliers_vector = Variable(n_steps + 1, nonneg=True)
    if truncated_mantissa:
        mantissa_vector = Variable(n_steps + 1, nonneg=True)

    dim = 2 * (n_steps + 1)

    # Create a matrix for each multiplier in rate_vector
    big_matrices_rate = [np.zeros((dim, dim)) for _ in range(n_steps + 1)]
    
    # Create a matrix for each multiplier in multipliers_vector
    big_matrices_multipliers = [[np.zeros((dim, dim)) for _ in range(n_steps + 1)] for _ in range(n_steps + 1)]
    
    # Create a matrix for each multiplier in inexact_multipliers_vector
    big_matrices_inexact = [np.zeros((dim, dim)) for _ in range(n_steps + 1)]

    if truncated_mantissa:
        big_matrices_mantissa = [np.zeros((dim, dim)) for _ in range(n_steps + 1)]

    # Fill in the matrices
    for i in range(n_steps + 1):
        # Contribution from inexact multipliers
        big_matrices_inexact[i][i, i] = -delta**2
        big_matrices_inexact[i][n_steps+1+i, n_steps+1+i] = 1

        # Contribution from rate multipliers
        big_matrices_rate[i][i, i] = -1

        if truncated_mantissa:
            big_matrices_mantissa[i][n_steps+1+i, n_steps+1+i]=1
            big_matrices_mantissa[i][i, n_steps+1+i]=1
            big_matrices_mantissa[i][n_steps+1+i, i]=1

        for j in range(i, n_steps + 1):
            if i != j and j - i <= distance_max:
                # Contribution from multipliers
                big_matrices_multipliers[i][j][i, i] =  0.5
                big_matrices_multipliers[i][j][i, j] = -0.5
                big_matrices_multipliers[i][j][j, i] = -0.5
                big_matrices_multipliers[i][j][j, j] =  0.5

                #print(i,j)
                for k in range(i, j):
                    #print(h[k])
                    big_matrices_multipliers[i][j][j, k] +=  h[k]/2
                    big_matrices_multipliers[i][j][k, j] +=  h[k]/2
                    big_matrices_multipliers[i][j][j, n_steps+1+k] += h[k]/2
                    big_matrices_multipliers[i][j][n_steps+1+k, j] += h[k]/2

                big_matrices_multipliers[j][i][i, i] = 0.5
                big_matrices_multipliers[j][i][i, j] = -0.5
                big_matrices_multipliers[j][i][j, i] = -0.5
                big_matrices_multipliers[j][i][j, j] = 0.5

                for k in range(i, j):
                    big_matrices_multipliers[j][i][i, k] += -h[k]/2
                    big_matrices_multipliers[j][i][k, i] += -h[k]/2
                    big_matrices_multipliers[j][i][i, n_steps+1+k] += -h[k]/2
                    big_matrices_multipliers[j][i][n_steps+1+k, i] += -h[k]/2

    big_matrix = sum([rate_vector[i] * big_matrices_rate[i] for i in range(n_steps + 1)])
    big_matrix += sum([inexact_multipliers_vector[i] * big_matrices_inexact[i] for i in range(n_steps + 1)])
    big_matrix += sum([multipliers_vector[i, j] * big_matrices_multipliers[i][j] 
                   for i in range(n_steps + 1) for j in range(n_steps + 1)])
    
    if truncated_mantissa:
        big_matrix += sum([mantissa_vector[i] * big_matrices_mantissa[i] for i in range(n_steps + 1)])

    #print(big_matrices_multipliers[2][1])
    # Constraints
    constraints = []
    constraints.append(
        sum(multipliers_vector[0, :]) - sum(multipliers_vector[:, 0]) == 1)
        
    constraints.append(
        sum(multipliers_vector[n_steps, :]) - sum(multipliers_vector[:, n_steps]) == -1,
    )
    for i in range(1, n_steps):
        constraints.append(sum(multipliers_vector[i, :]) - sum(multipliers_vector[:, i]) == 0)

    for i in range(n_steps + 1):
        constraints.append(multipliers_vector[i,i] == 0)
        for j in range(i, n_steps + 1):
            if j-i > distance_max:
                constraints.append(multipliers_vector[i,j] == 0)
                constraints.append(multipliers_vector[j,i] == 0)

    # Enforce positive semidefiniteness
    constraints.append(PSD(big_matrix))

    if metric == "last":
        prob = Problem(Maximize(rate_vector[-1]), constraints)
    if metric == "min":
        prob = Problem(Maximize(sum(rate_vector[:])), constraints)
    prob.solve(solver='MOSEK')

    return prob.status, multipliers_vector.value, inexact_multipliers_vector.value, rate_vector.value


def PEP_exact(h, print_=False, n_steps=1, distance_max=np.inf, metric = "min"):
    try:
        h[0]
        if len(h) != n_steps:
            print("n_steps is different from the size of list of steps")
    except:
        h = h*np.ones(n_steps)
    
    if metric == 'min':
        rate_vector = Variable(n_steps + 1, nonneg=True) #I changed here for several iter
    elif metric == 'min_non_neg':
        rate_vector = Variable(n_steps + 1, nonneg=True)
    else:
        rate_vector = Variable(n_steps + 1, nonneg=True)
    multipliers_vector = Variable((n_steps + 1, n_steps + 1), nonneg=True)

    dim = (n_steps + 1)

    # Create a matrix for each multiplier in rate_vector
    big_matrices_rate = [np.zeros((dim, dim)) for _ in range(n_steps + 1)]
    
    # Create a matrix for each multiplier in multipliers_vector
    big_matrices_multipliers = [[np.zeros((dim, dim)) for _ in range(n_steps + 1)] for _ in range(n_steps + 1)]

    # Fill in the matrices
    for i in range(n_steps + 1):
        # Contribution from rate multipliers
        big_matrices_rate[i][i, i] = -1

        for j in range(i, n_steps + 1):
            if i != j and j - i <= distance_max:
                # Contribution from multipliers
                big_matrices_multipliers[i][j][i, i] =  0.5
                big_matrices_multipliers[i][j][i, j] = -0.5
                big_matrices_multipliers[i][j][j, i] = -0.5
                big_matrices_multipliers[i][j][j, j] =  0.5

                for k in range(i, j):
                    big_matrices_multipliers[i][j][j, k] +=  h[k]/2
                    big_matrices_multipliers[i][j][k, j] +=  h[k]/2

                big_matrices_multipliers[j][i][i, i] = 0.5
                big_matrices_multipliers[j][i][i, j] = -0.5
                big_matrices_multipliers[j][i][j, i] = -0.5
                big_matrices_multipliers[j][i][j, j] = 0.5

                for k in range(i, j):
                    big_matrices_multipliers[j][i][i,k] += -h[k]/2
                    big_matrices_multipliers[j][i][k, i] += -h[k]/2

    big_matrix = sum([rate_vector[i] * big_matrices_rate[i] for i in range(n_steps + 1)])
    big_matrix += sum([multipliers_vector[i, j] * big_matrices_multipliers[i][j] 
                   for i in range(n_steps + 1) for j in range(n_steps + 1)])

    # Constraints
    constraints = []
    constraints.append(
        sum(multipliers_vector[0, :]) - sum(multipliers_vector[:, 0]) == 1)
        
    constraints.append(
        sum(multipliers_vector[n_steps, :]) - sum(multipliers_vector[:, n_steps]) == -1,
    )
    for i in range(1, n_steps):
        constraints.append(sum(multipliers_vector[i, :]) - sum(multipliers_vector[:, i]) == 0)

    for i in range(n_steps + 1):
        constraints.append(multipliers_vector[i,i] == 0)
        for j in range(i, n_steps + 1):
            if j-i > distance_max:
                constraints.append(multipliers_vector[i,j] == 0)
                constraints.append(multipliers_vector[j,i] == 0)

    constraints.append(PSD(big_matrix))

    if metric == "last":
        prob = Problem(Maximize(rate_vector[-1]), constraints)
    if metric == "min":
        prob = Problem(Maximize(sum(rate_vector[:])), constraints)
    if metric == 'min_non_neg':
        constraints.append(rate_vector[0]==0)
        prob = Problem(Maximize(sum(rate_vector[:])), constraints)
    prob.solve(solver='MOSEK')

    return prob.status, multipliers_vector.value, rate_vector.value


def eq_on_lambda(l,d,h):
    return ((d**2-1)*h**2+2*h)*l**3 + (2*(d**2-1)*h**2+5*h-4)*l**2 + ((d**2-1)*h**2+4*h-4)*l+h-1

def rate_in_right_sense(l,h):
    return (2*l)/(h*l**2+2*(h-1)*l+h-1)

def rate_middle_numeric(l, fixed_d):
    h_list_test = np.arange((1.5)/(fixed_d+1), (3*fixed_d+2-np.sqrt(4-3*fixed_d**2))/(2*fixed_d*(fixed_d+1))+0.001,0.001)
    # h_list_test = np.arange(0.001,2/(1+fixed_d), 0.001)
    rate_explicit = []
    lambda_value = []
    for i in range(len(h_list_test)):
        sol = sym.solve(eq_on_lambda(l,fixed_d,h_list_test[i]))
        sol_real = []
        for j in range(len(sol)):
            sol_real.append(sym.re(sol[j]))
        lambda_value.append(np.max(np.array(sol_real)))
        rate_explicit.append(rate_in_right_sense(np.max(np.array(sol_real)),h_list_test[i]))
    return h_list_test, np.array(rate_explicit), np.array(lambda_value)
