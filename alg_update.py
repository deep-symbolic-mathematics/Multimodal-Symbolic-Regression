import numpy as np

def lso_update_pop(
    pop, fitness, iteration, Max_iter, 
    Alpha_score, Beta_score, Delta_score ,
    Alpha_pos, Beta_pos, Delta_pos,
    lb=-5, ub=5
    ):
    Positions = pop.cpu().numpy()

    # Return back the search agents that go beyond the boundaries of the search space
    Positions = np.clip(Positions, lb, ub)

    N = Positions.shape[0]

    # Combine Positions and fitness separately
    combined_positions = np.concatenate((Positions, Alpha_pos[np.newaxis, :], Beta_pos[np.newaxis, :], Delta_pos[np.newaxis, :]), axis=0)
    combined_fitness = np.concatenate((fitness, np.array([Alpha_score]), np.array([Beta_score]), np.array([Delta_score])))

    # Find the indices of the best three positions (minimum fitness values)
    best_indices = np.argsort(combined_fitness)[:3]
    Alpha_index, Beta_index, Delta_index = best_indices[0], best_indices[1], best_indices[2]

    # Update the Alpha_pos, Alpha_score, Beta_pos, Beta_score, Delta_pos, and Delta_score
    Alpha_pos = combined_positions[Alpha_index]
    Alpha_score = combined_fitness[Alpha_index]
    Beta_pos = combined_positions[Beta_index]
    Beta_score = combined_fitness[Beta_index]
    Delta_pos = combined_positions[Delta_index]
    Delta_score = combined_fitness[Delta_index]

    a = 2 - iteration * ((2) / Max_iter)
    # a decreases linearly from 2 to 0

    r1 = np.random.random((len(Positions), Positions.shape[1]))  # Generate random numbers in [0,1]
    r2 = np.random.random((len(Positions), Positions.shape[1])) 
    A1 = 2 * a * r1 - a
    C1 = 2 * r2
    D_alpha = np.abs(C1 * Alpha_pos.reshape(1, -1) - Positions)
    X1 = Alpha_pos.reshape(1, -1) - A1 * D_alpha

    r1 = np.random.random((len(Positions), Positions.shape[1]))
    r2 = np.random.random((len(Positions), Positions.shape[1]))
    A2 = 2 * a * r1 - a
    C2 = 2 * r2
    D_beta = np.abs(C2 * Beta_pos.reshape(1, -1) - Positions)
    X2 = Beta_pos.reshape(1, -1) - A2 * D_beta

    r1 = np.random.random((len(Positions), Positions.shape[1]))
    r2 = np.random.random((len(Positions), Positions.shape[1]))
    A3 = 2 * a * r1 - a
    C3 = 2 * r2
    D_delta = np.abs(C3 * Delta_pos.reshape(1, -1) - Positions)
    X3 = Delta_pos.reshape(1, -1) - A3 * D_delta

    Prev_Positions = Positions.copy()
    Positions = (X1 + X2 + X3) / 3

    elites = Alpha_score, Beta_score, Delta_score , Alpha_pos, Beta_pos, Delta_pos
    return Positions , elites


def perturb_update_pop(
    pop, original_pop, lb=-5, ub=5):
    Positions = pop.cpu().numpy()
    Positions = np.clip(Positions, lb, ub)
    N = Positions.shape[0]

    min_perturb_std_factor = 0.01
    max_perturb_std_factor = 1

    perturbed_positions = np.copy(Positions)

    for i in range(Positions.shape[0]):
        sample_vector = Positions[i, :]
        vector_std = np.std(sample_vector)

        # Sample perturbation standard deviation from a uniform distribution
        perturb_std = np.random.uniform(
            vector_std * min_perturb_std_factor,
            vector_std * max_perturb_std_factor
        )

        # Perturb the sample vector with Gaussian noise
        perturbation = np.random.normal(0, perturb_std, size=sample_vector.shape)
        perturbed_positions[i, :] += perturbation

    # breakpoint()
    return Positions 


