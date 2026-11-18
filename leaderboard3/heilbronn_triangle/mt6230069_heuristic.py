# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def is_point_in_triangle(point, tol=1e-6):
    """Check if a point is inside or on the boundary of the triangle."""
    x, y = point
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Check constraints:
    # y >= 0
    # y <= sqrt(3) * x  
    # y <= sqrt(3) * (1 - x)
    sqrt3 = np.sqrt(3)
    
    if y < -tol:
        return False
    if y > sqrt3 * x + tol:
        return False
    if y > sqrt3 * (1 - x) + tol:
        return False
    return True


def triangle_area(p1, p2, p3):
    """Calculate the area of triangle formed by three points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Using cross product formula
    return 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))


def min_triangle_area(points):
    """Calculate the minimum triangle area over all combinations of three points."""
    n = len(points)
    if n < 3:
        return 0
    
    min_area = float('inf')
    
    # Check all combinations of 3 points
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = triangle_area(points[i], points[j], points[k])
                min_area = min(min_area, area)
                
    return min_area


def project_point_to_triangle(point):
    """Project a point to the nearest point in the triangle."""
    x, y = point
    sqrt3 = np.sqrt(3)
    
    # Clip coordinates to triangle bounds
    # y should be between 0 and min(sqrt3*x, sqrt3*(1-x))
    y_max = min(sqrt3 * x, sqrt3 * (1 - x))
    y = max(0, min(y, y_max))
    x = max(0, min(x, 1))
    
    # Additional correction to ensure it's truly in triangle
    if y > sqrt3 * x:
        # Point is above the right edge, project to that edge
        y = sqrt3 * x
    if y > sqrt3 * (1 - x):
        # Point is above the left edge, project to that edge
        y = sqrt3 * (1 - x)
        
    return np.array([x, y])


def generate_structured_points():
    """Generate a structured initial configuration of points."""
    # Create a structured arrangement that tends to avoid degenerate cases
    # Place points on a grid pattern but slightly perturbed to avoid exact alignments
    
    # Edge points (some on vertices, some on edges)
    edge_points = [
        [0, 0],           # Vertex
        [1, 0],           # Vertex  
        [0.5, np.sqrt(3)/2],  # Vertex
        [0.25, 0],        # Edge point
        [0.75, 0],        # Edge point
        [0.1, np.sqrt(3)*0.1],  # Interior edge point
        [0.9, np.sqrt(3)*0.1],  # Interior edge point
        [0.5, np.sqrt(3)*0.2],  # Middle edge point
    ]
    
    # Interior points
    interior_points = [
        [0.3, np.sqrt(3)*0.3],
        [0.7, np.sqrt(3)*0.3],
        [0.5, np.sqrt(3)*0.5],
    ]
    
    # Combine all points
    points = edge_points + interior_points
    
    # Make sure we have exactly 11 points
    while len(points) < 11:
        # Add random points but ensure they're in triangle
        while True:
            x = np.random.random()
            y = np.random.random() * np.sqrt(3) / 2
            if y <= np.sqrt(3) * x and y <= np.sqrt(3) * (1 - x):
                points.append([x, y])
                break
    
    # Ensure exactly 11 points and project to triangle
    points = points[:11]
    for i in range(len(points)):
        points[i] = project_point_to_triangle(np.array(points[i]))
    
    return np.array(points)


def objective_with_constraints(x_flat, points_count, sqrt3):
    """Objective function for optimization with constraints"""
    # Reshape flat array to points
    points = x_flat.reshape((points_count, 2))
    
    # Calculate minimum triangle area
    min_area = min_triangle_area(points)
    
    # We want to maximize the minimum area, so we minimize the negative of it
    return -min_area


def constraint_func(x_flat, points_count, sqrt3):
    """Constraint function to ensure all points stay within triangle."""
    points = x_flat.reshape((points_count, 2))
    
    # Constraint: points must be within triangle
    constraints = []
    for i in range(points_count):
        x, y = points[i]
        # y >= 0
        constraints.append(y)
        # y <= sqrt(3) * x  
        constraints.append(sqrt3 * x - y)
        # y <= sqrt(3) * (1 - x)
        constraints.append(sqrt3 * (1 - x) - y)
    
    return np.array(constraints)


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    np.random.seed(0)  # For reproducibility
    start_time = time.perf_counter()
    deadline = start_time + 350  # Leave some buffer for cleanup
    
    sqrt3 = np.sqrt(3)
    points_count = 11
    
    # Start with structured initialization
    best_points = generate_structured_points()
    best_min_area = min_triangle_area(best_points)
    
    # Use SLSQP optimization with MANY restarts to improve solution
    total_restarts = 50  # Increased from 10 to 50
    
    # Adaptive restart strategy - use more iterations in later phases
    for restart in range(total_restarts):  
        if time.perf_counter() > deadline - 5:
            break
            
        # Determine adaptive parameters based on restart number
        if restart < 10:
            # Early restarts: aggressive perturbation, fewer iterations
            perturbation_magnitude = 0.02
            maxiter = 50
        elif restart < 30:
            # Mid restarts: moderate perturbation, moderate iterations
            perturbation_magnitude = 0.01
            maxiter = 100
        else:
            # Later restarts: fine perturbation, many iterations for precision
            perturbation_magnitude = 0.005
            maxiter = 200
        
        # Start from a slightly perturbed version of current best
        current_points = best_points.copy()
        # Add adaptive perturbation to avoid stagnation
        for i in range(len(current_points)):
            current_points[i] += np.random.normal(0, perturbation_magnitude, 2)
            current_points[i] = project_point_to_triangle(current_points[i])
        
        # Flatten for optimization
        x0 = current_points.flatten()
        
        # Define bounds for each coordinate (0 to 1 for x, 0 to sqrt(3)/2 for y)
        bounds = []
        for i in range(points_count):
            bounds.extend([(0, 1), (0, sqrt3/2)])
        
        # Define constraints
        def ineq_con(x):
            return constraint_func(x, points_count, sqrt3)
        
        constraints = {'type': 'ineq', 'fun': ineq_con}
        
        try:
            # Optimize using SLSQP with adaptive parameters
            result = minimize(
                objective_with_constraints,
                x0,
                args=(points_count, sqrt3),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-10},
                callback=lambda x: None  # No callback needed
            )
            
            if result.success:
                optimized_points = result.x.reshape((points_count, 2))
                # Project back to triangle to make sure they're valid
                for i in range(len(optimized_points)):
                    optimized_points[i] = project_point_to_triangle(optimized_points[i])
                
                new_min_area = min_triangle_area(optimized_points)
                if new_min_area > best_min_area:
                    best_min_area = new_min_area
                    best_points = optimized_points.copy()
                    
        except Exception:
            # If optimization fails, continue with current best
            pass
            
        # Occasionally try a completely new structure with increased randomness
        if restart % 5 == 0 and restart > 0:
            new_structured = generate_structured_points()
            # Add even more randomness to the new structure
            for i in range(len(new_structured)):
                new_structured[i] += np.random.normal(0, 0.01, 2)
                new_structured[i] = project_point_to_triangle(new_structured[i])
            
            new_area = min_triangle_area(new_structured)
            if new_area > best_min_area:
                best_min_area = new_area
                best_points = new_structured.copy()
    
    # Final validation
    for i in range(len(best_points)):
        if not is_point_in_triangle(best_points[i]):
            # If any point is not in triangle, generate new structured points
            return generate_structured_points()
    
    # Ensure we have 11 points
    if len(best_points) != 11:
        return generate_structured_points()
    
    return best_points


# EVOLVE-BLOCK-END
