# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Deadline for computation
    t0 = time.perf_counter()
    deadline = t0 + 360.0
    
    # Initialize with a reasonable configuration
    n = 21
    best_circles = None
    best_sum = 0
    
    # SDP-based construction with multiple restarts
    # Extend construction phase to 340s and reduce refinement to 20s
    max_restarts = 100  # Increased number of restarts to compensate for longer construction time
    previous_best = None
    
    for restart in range(max_restarts):
        if time.perf_counter() > deadline:
            break
            
        # Generate random initial configuration with better spatial distribution
        circles = np.zeros((n, 3))
        
        # Use systematic initialization with hexagonal grid pattern
        if restart == 0:
            # First restart: use hexagonal pattern
            rows = 5
            cols = 5
            spacing_x = 1.0 / cols
            spacing_y = 1.0 / rows
            
            # Create hexagonal grid with slight randomness
            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n:
                        break
                    x = (j + 0.5) * spacing_x
                    y = (i + 0.5) * spacing_y
                    # Add some randomness
                    x += np.random.normal(0, 0.02)
                    y += np.random.normal(0, 0.02)
                    # Clip to valid range
                    x = np.clip(x, 0.01, 0.99)
                    y = np.clip(y, 0.01, 0.99)
                    
                    # Estimate appropriate radius based on spacing
                    max_radius = min(spacing_x, spacing_y) / 2.0 * 0.8
                    
                    # Add some randomness to radius too
                    radius = max_radius * (0.7 + 0.6 * np.random.random())
                    
                    # Make sure it's valid
                    if radius > 0 and x - radius > 0 and x + radius < 1 and y - radius > 0 and y + radius < 1:
                        circles[idx] = [x, y, radius]
                        idx += 1
                        
            # Fill any unfilled positions
            while idx < n:
                x = np.random.uniform(0.01, 0.99)
                y = np.random.uniform(0.01, 0.99)
                radius = np.random.uniform(0.01, 0.1)
                if x - radius > 0 and x + radius < 1 and y - radius > 0 and y + radius < 1:
                    circles[idx] = [x, y, radius]
                    idx += 1
                    
        else:
            # Use warm-start initialization from previous best if available
            if previous_best is not None and restart > 0:
                # Perturb the previous best solution slightly
                circles = previous_best.copy()
                for i in range(n):
                    # Add small random perturbation to position
                    circles[i, 0] += np.random.normal(0, 0.01)
                    circles[i, 1] += np.random.normal(0, 0.01)
                    # Keep radii roughly the same but allow small changes
                    circles[i, 2] *= (0.95 + np.random.random() * 0.1)
                    # Clamp to valid ranges
                    circles[i, 0] = np.clip(circles[i, 0], 0.01, 0.99)
                    circles[i, 1] = np.clip(circles[i, 1], 0.01, 0.99)
                    circles[i, 2] = np.clip(circles[i, 2], 0.001, 0.49)
            else:
                # Create initial configuration - place circles randomly in a 1x1 square
                for i in range(n):
                    circles[i] = [
                        np.random.uniform(0.01, 0.99),
                        np.random.uniform(0.01, 0.99),
                        np.random.uniform(0.01, 0.1)
                    ]
        
        # Try to optimize this configuration through proper overlap resolution
        optimized_circles = optimize_circles_convex(circles, deadline)
        
        if optimized_circles is not None:
            # Validate this configuration
            if is_valid_packing(optimized_circles):
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles.copy()
                    previous_best = optimized_circles.copy()
    
    # If we didn't find anything, return a fallback
    if best_circles is None:
        # Return a basic arrangement that satisfies constraints
        best_circles = np.zeros((n, 3))
        
        # Simple grid arrangement with decreasing sizes
        rows = 3
        cols = 7
        width = 1.0
        height = 1.0
        
        cell_width = width / cols
        cell_height = height / rows
        
        # Place circles in a grid pattern
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 0.5) * cell_width
                y = (i + 0.5) * cell_height
                
                # Radius based on how much space we have and spacing
                max_radius = min(cell_width, cell_height) / 2.0
                # Decrease radii for later circles
                radius = max_radius * (0.8 + 0.2 * (n - idx) / n)
                
                # Make sure we don't exceed bounds  
                if x - radius < 0 or x + radius > width or y - radius < 0 or y + radius > height:
                    radius = min(x, width - x, y, height - y)
                
                if radius > 0:
                    best_circles[idx] = [x, y, radius]
                    idx += 1
                    
                if idx >= n:
                    break
        # Ensure we have exactly n circles
        for i in range(idx, n):
            best_circles[i] = [0.5, 0.5, 0.01]  # Fallback small circles
        
    return best_circles


def optimize_circles_convex(initial_circles, deadline):
    """Optimize circle positions and radii using convex optimization approach"""
    n = len(initial_circles)
    
    # Flatten the problem into variables: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    def flatten_circles(circles):
        return np.array([coord for circle in circles for coord in circle])
    
    def unflatten_circles(flattened):
        return flattened.reshape(-1, 3)
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(flattened_circles):
        circles = unflatten_circles(flattened_circles)
        return -np.sum(circles[:, 2])
    
    # Constraints for overlapping circles
    def overlap_constraints(flattened_circles):
        circles = unflatten_circles(flattened_circles)
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                # Constraint: dist >= r1 + r2 (so we don't overlap)  
                # We'll use dist >= r1 + r2 - 1e-6 for numerical stability
                constraints.append(dist - (r1 + r2) + 1e-6)
        return np.array(constraints)
    
    # Boundary constraints: circles must fit within [0,1]x[0,1]
    def boundary_constraints(flattened_circles):
        circles = unflatten_circles(flattened_circles)
        constraints = []
        # Each circle center must be within bounds accounting for radius
        for i in range(n):
            x, y, r = circles[i]
            # x - r >= 0
            constraints.append(x - r)
            # y - r >= 0  
            constraints.append(y - r)
            # x + r <= 1
            constraints.append(1 - x - r)
            # y + r <= 1
            constraints.append(1 - y - r)
        return np.array(constraints)
    
    # Create bounds for variables: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    # x, y: [0.001, 0.999] (slightly inside boundaries to avoid edge cases)
    # r: [0.001, 0.499] (bounded to prevent extreme values)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
    
    # Initial guess
    x0 = flatten_circles(initial_circles)
    
    # Define constraints properly
    constraints = []
    
    # Overlap constraints (inequality: g(x) <= 0 means g(x) >= 0)
    overlap_constraint = {
        'type': 'ineq',
        'fun': lambda x: overlap_constraints(x)
    }
    constraints.append(overlap_constraint)
    
    # Boundary constraints
    boundary_constraint = {
        'type': 'ineq', 
        'fun': lambda x: boundary_constraints(x)
    }
    constraints.append(boundary_constraint)
    
    # Use SLSQP method which handles both constraints and bounds
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-6},
            callback=lambda x: None  # We don't need a callback
        )
        
        if result.success:
            optimized_circles = unflatten_circles(result.x)
            # Verify that this is a valid solution
            if is_valid_packing(optimized_circles):
                return optimized_circles
    except Exception:
        # If optimization fails, fall back to heuristic
        pass
    
    # Fall back to heuristic approach if optimization doesn't work
    return optimize_circles_heuristic(initial_circles, deadline)


def optimize_circles_heuristic(initial_circles, deadline):
    """Optimize circle positions using systematic boundary-aware radius expansion"""
    n = len(initial_circles)
    
    # Start with initial configuration
    circles = initial_circles.copy()
    
    # Systematic approach: first resolve overlaps, then expand radii systematically
    max_iterations = 300  # Reduced to allow more time for better expansion
    for iteration in range(max_iterations):
        if time.perf_counter() > deadline:
            break
            
        # Find all overlaps with detailed information
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < r1 + r2 - 1e-6:
                    overlap_amount = (r1 + r2 - dist)
                    overlaps.append((i, j, overlap_amount, dx, dy, dist))
        
        if not overlaps:
            # Now focus on systematic radius expansion based on boundary contact and geometry
            # Prioritize expanding circles that are near boundaries or have boundary contact
            expanded = False
            # Calculate boundary distances for all circles
            boundary_distances = []
            for i in range(n):
                x, y, r = circles[i]
                dist_to_boundary = min(x, 1.0-x, y, 1.0-y)
                boundary_distances.append(dist_to_boundary)
            
            # Sort circles by how close they are to boundaries (closest first)
            # These are the circles most likely to benefit from expansion
            sorted_by_boundary = sorted(range(n), key=lambda i: boundary_distances[i])
            
            # Try to expand each circle from closest to boundary to farthest
            for i in sorted_by_boundary:
                old_radius = circles[i, 2]
                
                # Calculate the maximum radius this circle can have
                # Considering both boundary constraints and neighbor constraints
                max_radius = 0.49  # Upper bound
                
                # Boundary constraint
                boundary_constraint = min(
                    circles[i, 0], 
                    1.0 - circles[i, 0], 
                    circles[i, 1], 
                    1.0 - circles[i, 1]
                ) - 1e-6
                
                max_radius = min(max_radius, boundary_constraint)
                
                # Neighbor constraints
                for j in range(n):
                    if i != j:
                        dx = circles[i, 0] - circles[j, 0]
                        dy = circles[i, 1] - circles[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        # Must leave room for the other circle's radius and a small gap
                        neighbor_constraint = dist - circles[j, 2] - 1e-6
                        max_radius = min(max_radius, neighbor_constraint)
                
                if max_radius > old_radius + 1e-6:
                    # Try to expand radius to maximum possible
                    new_radius = max_radius
                    
                    # Verify this doesn't cause overlaps
                    temp_circles = circles.copy()
                    temp_circles[i, 2] = new_radius
                    
                    # Check if expanding this radius breaks any other constraints
                    valid = True
                    for k in range(n):
                        if k != i:
                            dx = temp_circles[k, 0] - temp_circles[i, 0]
                            dy = temp_circles[k, 1] - temp_circles[i, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            # Should be at least the sum of radii minus tiny tolerance
                            if dist < temp_circles[k, 2] + temp_circles[i, 2] - 1e-6:
                                valid = False
                                break
                    
                    if valid and is_valid_packing(temp_circles):
                        circles = temp_circles
                        expanded = True
            
            # If we made no expansions, we're done
            if not expanded:
                break
        else:
            # Resolve overlaps systematically with a better approach
            # Sort by overlap severity (largest first) for more aggressive resolution
            overlaps.sort(key=lambda x: x[2], reverse=True)
            
            # Handle overlaps by moving circles away from each other with priority to boundary contact
            for i, j, overlap_amount, dx, dy, dist in overlaps:
                if time.perf_counter() > deadline:
                    break
                    
                if dist > 1e-10:
                    # Prioritize separating circles based on which ones are closer to boundaries
                    # This encourages movement toward expanding contact with boundaries
                    i_boundary_dist = min(circles[i, 0], 1.0 - circles[i, 0], 
                                        circles[i, 1], 1.0 - circles[i, 1])
                    j_boundary_dist = min(circles[j, 0], 1.0 - circles[j, 0], 
                                        circles[j, 1], 1.0 - circles[j, 1])
                    
                    # Weight towards more boundary-constrained circles
                    i_weight = max(0.3, 1.0 - i_boundary_dist * 3.0)  # Higher weight for closer to boundary
                    j_weight = max(0.3, 1.0 - j_boundary_dist * 3.0)
                    
                    # Normalize weights based on total radii
                    total_radii = circles[i, 2] + circles[j, 2]
                    if total_radii > 0:
                        i_weight *= circles[i, 2] / total_radii
                        j_weight *= circles[j, 2] / total_radii
                    else:
                        i_weight = 0.5
                        j_weight = 0.5
                    
                    factor = overlap_amount / (dist + 1e-12) * 0.3
                    
                    # Apply weighted separation
                    circles[i, 0] -= dx * factor * i_weight
                    circles[i, 1] -= dy * factor * i_weight
                    circles[j, 0] += dx * factor * j_weight
                    circles[j, 1] += dy * factor * j_weight
                else:
                    # Random small separation if at same position
                    angle = np.random.random() * 2 * np.pi
                    circles[i, 0] -= np.cos(angle) * 0.001
                    circles[i, 1] -= np.sin(angle) * 0.001
                    circles[j, 0] += np.cos(angle) * 0.001
                    circles[j, 1] += np.sin(angle) * 0.001
            
            # Apply strict boundary corrections
            for i in range(n):
                x, y, r = circles[i]
                circles[i, 0] = np.clip(x, r + 1e-6, 1.0 - r - 1e-6)
                circles[i, 1] = np.clip(y, r + 1e-6, 1.0 - r - 1e-6)
    
    # Additional focused radius expansion phase
    # Try more comprehensive boundary-contact-based expansion
    max_radius_expansion_iterations = 100
    for iteration in range(max_radius_expansion_iterations):
        if time.perf_counter() > deadline:
            break
            
        # Focus on expanding circles that are already touching boundaries
        # These are the prime candidates for maximizing total area
        expanded = False
        
        # Check all circles for potential expansion
        for i in range(n):
            old_radius = circles[i, 2]
            
            # Calculate constraints for expansion
            max_radius = 0.49
            
            # Boundary constraints
            boundary_constraint = min(
                circles[i, 0], 
                1.0 - circles[i, 0], 
                circles[i, 1], 
                1.0 - circles[i, 1]
            ) - 1e-6
            
            max_radius = min(max_radius, boundary_constraint)
            
            # Neighbor constraints
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    neighbor_constraint = dist - circles[j, 2] - 1e-6
                    max_radius = min(max_radius, neighbor_constraint)
            
            if max_radius > old_radius + 1e-6:
                # Try to expand to max radius
                new_radius = max_radius
                
                temp_circles = circles.copy()
                temp_circles[i, 2] = new_radius
                
                # Validate this new configuration
                valid = True
                for k in range(n):
                    if k != i:
                        dx = temp_circles[k, 0] - temp_circles[i, 0]
                        dy = temp_circles[k, 1] - temp_circles[i, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < temp_circles[k, 2] + temp_circles[i, 2] - 1e-6:
                            valid = False
                            break
                
                if valid and is_valid_packing(temp_circles):
                    circles = temp_circles
                    expanded = True
        
        if not expanded:
            break
    
    # Final validation and adjustment
    if is_valid_packing(circles):
        return circles
    else:
        # If still invalid, try to create a valid configuration using better refinement
        return refine_initial_placement_simple(initial_circles)


def refine_initial_placement_simple(initial_circles):
    """A simpler but more robust refinement of initial placement"""
    n = len(initial_circles)
    circles = initial_circles.copy()
    
    # Try to fix overlap issues through systematic correction
    max_iterations = 500
    for _ in range(max_iterations):
        moved = False
        # Find all overlaps
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < r1 + r2 - 1e-6:
                    overlap_amount = (r1 + r2 - dist)
                    overlaps.append((i, j, overlap_amount, dx, dy, dist))
        
        if not overlaps:
            break
            
        # Resolve overlaps systematically
        for i, j, overlap_amount, dx, dy, dist in overlaps:
            if dist > 1e-10:
                total_radius = circles[i, 2] + circles[j, 2]
                if total_radius > 0:
                    factor = overlap_amount / (dist + 1e-12) * 0.3
                    circles[i, 0] -= dx * factor * (circles[i, 2] / total_radius)
                    circles[i, 1] -= dy * factor * (circles[i, 2] / total_radius)
                    circles[j, 0] += dx * factor * (circles[j, 2] / total_radius)
                    circles[j, 1] += dy * factor * (circles[j, 2] / total_radius)
            else:
                # Random small separation if at same position
                angle = np.random.random() * 2 * np.pi
                circles[i, 0] -= np.cos(angle) * 0.001
                circles[i, 1] -= np.sin(angle) * 0.001
                circles[j, 0] += np.cos(angle) * 0.001
                circles[j, 1] += np.sin(angle) * 0.001
            moved = True
            
        # Apply boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            circles[i, 0] = np.clip(x, r + 1e-6, 1.0 - r - 1e-6)
            circles[i, 1] = np.clip(y, r + 1e-6, 1.0 - r - 1e-6)
            
        if not moved:
            break
            
    return circles


def is_valid_packing(circles):
    """Check if the circle packing is valid"""
    # Check for NaN or negative radii
    if np.any(np.isnan(circles)) or np.any(circles[:, 2] < 0):
        return False
    
    # Check overlap
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            distance = np.sqrt(dx * dx + dy * dy)
            min_distance = circles[i, 2] + circles[j, 2]
            if distance < min_distance - 1e-6:
                return False
    
    # Check boundary constraints
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width + height > 2.0 + 1e-6:
        return False
    
    return True


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
