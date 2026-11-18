GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


import time
import random
import math

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses greedy assignment followed by simulated annealing to improve load balance.
    Implements Iterative Load Balancing with Simulated Annealing (ILB-SA).
    """

    # Helper to calculate KVPR for a given placement
    def calculate_kvpr(placement):
        max_kvpr = 0.0
        for gpu_id, gpu_models in placement.items():
            if not gpu_models:
                continue
            total_model_size = sum(m.model_size for m in gpu_models)
            if total_model_size >= GPU_MEM_SIZE:
                return float('inf')
            total_req_rate_slo = sum(m.req_rate / m.slo for m in gpu_models)
            kvpr = total_req_rate_slo / (GPU_MEM_SIZE - total_model_size)
            max_kvpr = max(max_kvpr, kvpr)
        return max_kvpr

    # Bin-packing inspired initial placement with load balancing awareness
    def bin_packing_placement(models_list):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        gpu_loads = [0.0 for _ in range(gpu_num)]  # Track total model sizes per GPU
        gpu_kvpr = [0.0 for _ in range(gpu_num)]   # Track weighted request rates per GPU

        # Create list of (priority_score, model) tuples and sort by priority descending
        # Priority: req_rate / slo (higher is better)
        model_priority_pairs = [(m.req_rate / m.slo, m) for m in models_list]
        model_priority_pairs.sort(key=lambda x: x[0], reverse=True)
        sorted_models = [m for _, m in model_priority_pairs]

        for model in sorted_models:
            # Find the GPU with lowest current KVPR that can accommodate this model
            best_gpu = None
            best_kvpr = float('inf')
            
            for gpu_id in range(gpu_num):
                if model.model_size <= GPU_MEM_SIZE - gpu_loads[gpu_id]:
                    # Calculate the KVPR if we place this model on this GPU
                    current_total_req_rate_slo = gpu_kvpr[gpu_id] + model.req_rate / model.slo
                    current_total_model_size = gpu_loads[gpu_id] + model.model_size
                    if current_total_model_size < GPU_MEM_SIZE:
                        proposed_kvpr = current_total_req_rate_slo / (GPU_MEM_SIZE - current_total_model_size)
                    else:
                        proposed_kvpr = float('inf')
                        
                    # Prefer GPUs with lower KVPR
                    if proposed_kvpr < best_kvpr or best_gpu is None:
                        best_kvpr = proposed_kvpr
                        best_gpu = gpu_id

            # Place on the best GPU, even if it leads to high KVPR (better than error)
            if best_gpu is None:
                # Resort to any available GPU if needed (last resort fallback)
                for gpu_id in range(gpu_num):
                    if model.model_size <= GPU_MEM_SIZE - gpu_loads[gpu_id]:
                        best_gpu = gpu_id
                        break
                if best_gpu is None:
                    raise ValueError(
                        f"Unable to place model of size {model.model_size} GB on any GPU. "
                        f"Remaining per-GPU memory: {[GPU_MEM_SIZE - l for l in gpu_loads]}"
                    )

            # Add model to the chosen GPU
            placement[best_gpu].append(model)
            gpu_loads[best_gpu] += model.model_size
            gpu_kvpr[best_gpu] += model.req_rate / model.slo

        return placement

    # Apply a specified move
    def apply_move(placement, move_type, gpu_a, gpu_b, model_a, model_b=None):
        new_placement = {gpu_id: placement[gpu_id][:] for gpu_id in range(gpu_num)}
        
        if move_type == 'move':
            # Remove from source GPU
            try:
                idx = new_placement[gpu_a].index(model_a)
                new_placement[gpu_a].pop(idx)
            except ValueError:
                return None
            # Add to destination GPU
            new_placement[gpu_b].append(model_a)
            
        elif move_type == 'swap':
            try:
                idx_a = new_placement[gpu_a].index(model_a)
                idx_b = new_placement[gpu_b].index(model_b)
                new_placement[gpu_a][idx_a] = model_b
                new_placement[gpu_b][idx_b] = model_a
            except ValueError:
                return None
        else:
            return None
        
        return new_placement

    # Multi-start with diversified initial placements
    best_placement = None
    best_kvpr = float('inf')
    
    # Try multiple random restarts
    num_restarts = 5
    t0 = time.perf_counter()
    deadline = t0 + 2.57
    
    for restart in range(num_restarts):
        if time.perf_counter() >= deadline:
            break
            
        # Create a shuffled copy for this restart
        models_copy = models[:]
        random.shuffle(models_copy)
        
        # Perform bin-packing inspired placement
        placement = bin_packing_placement(models_copy)
        
        # Polish with simulated annealing
        current_placement = {gpu_id: placement[gpu_id][:] for gpu_id in range(gpu_num)}
        current_kvpr = calculate_kvpr(current_placement)
        
        # Simulated Annealing parameters
        temp_start = 1.0
        temp_min = 1e-6
        alpha = 0.95
        max_iterations = 5000
        
        local_deadline = min(deadline, t0 + 2.0)  # Leave time for final validation
        
        # Generate moves
        def generate_moves(placement, gpu_num):
            moves = []
            for gpu_a in range(gpu_num):
                if not placement[gpu_a]: continue
                for gpu_b in range(gpu_num):
                    if gpu_a == gpu_b: continue
                    for model in placement[gpu_a]:
                        # Move operation: transfer model from gpu_a to gpu_b
                        new_b_size = sum(m.model_size for m in placement[gpu_b]) + model.model_size
                        if new_b_size <= GPU_MEM_SIZE:
                            moves.append(('move', gpu_a, gpu_b, model))
                    
                    # Swap operations: between models in gpu_a and gpu_b
                    for model_a in placement[gpu_a]:
                        for model_b in placement[gpu_b]:
                            old_a_size = sum(m.model_size for m in placement[gpu_a])
                            old_b_size = sum(m.model_size for m in placement[gpu_b])
                            
                            new_a_size = old_a_size - model_a.model_size + model_b.model_size
                            new_b_size = old_b_size - model_b.model_size + model_a.model_size
                            
                            if new_a_size <= GPU_MEM_SIZE and new_b_size <= GPU_MEM_SIZE:
                                moves.append(('swap', gpu_a, gpu_b, model_a, model_b))
            return moves

        # Simulated Annealing main loop
        temp = temp_start
        iteration = 0
        
        while time.perf_counter() < local_deadline and iteration < max_iterations:
            # Generate all valid moves
            moves = generate_moves(current_placement, gpu_num)
            
            if not moves:
                break
                
            # Randomly select a move
            move = random.choice(moves)
            move_type, gpu_a, gpu_b = move[0], move[1], move[2]
            model_a = move[3]
            model_b = move[4] if len(move) > 4 else None

            # Apply move
            new_placement = apply_move(current_placement, move_type, gpu_a, gpu_b, model_a, model_b)
            if new_placement is None:
                continue
                
            # Calculate new KVPR
            new_kvpr = calculate_kvpr(new_placement)
            
            # Accept or reject the move based on simulated annealing
            delta = new_kvpr - current_kvpr
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_placement = new_placement
                current_kvpr = new_kvpr
            
            # Cool down the temperature
            temp = max(temp_min, temp * alpha)
            iteration += 1
                
        # Update global best if this restart produced better result
        if current_kvpr < best_kvpr:
            best_kvpr = current_kvpr
            best_placement = {gpu_id: current_placement[gpu_id][:] for gpu_id in range(gpu_num)}
    
    # If no restart succeeded, fall back to single bin-packing placement
    if best_placement is None:
        random.shuffle(models)
        placement = bin_packing_placement(models)
        best_placement = {gpu_id: placement[gpu_id][:] for gpu_id in range(gpu_num)}
    
    # Final validation - ensure all models are present and not duplicated
    all_placed = []
    for gpu_models in best_placement.values():
        all_placed.extend(gpu_models)
    
    placed_ids = set(id(m) for m in all_placed)
    input_ids = set(id(m) for m in models)
    
    if placed_ids != input_ids:
        # Fallback to basic bin-packing solution if validation fails
        # Need to be careful to avoid creating new model objects
        placement = bin_packing_placement(models)
        best_placement = {gpu_id: placement[gpu_id][:] for gpu_id in range(gpu_num)}
        
    return best_placement


# EVOLVE-BLOCK-END



