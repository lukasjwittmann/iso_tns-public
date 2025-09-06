"""Toy code for the optimal complexity of a tensor contraction in terms of symbolic dimensions."""

import numpy as np
import opt_einsum as oe


def get_single_complexity(einsum_equation, indices_to_symbols):
    """For a single tensor contraction described by einsum_equation, return the complexity equal to
    the dimension of all contracted indices times the dimension of all open indices. The dimension
    for each index is given symbolically with the dictionary indices_to_symbols.

    example: (AB)_acde = sum_b A_abc B_bde
             - einsum_equation = "abc,bde->acde"
             - indices_to_symbols = {"a": "M", "b": "M", "c": "N", "d": "K", "e": "K"}
             -> complexity of order (M^2)(N)(K^2)
    """
    input, output = einsum_equation.split("->")  # "abc,bde", "acde"
    inputs = input.split(",")  # ["abc", "bde"]
    assert len(inputs) == 2, f"expected 2 input tensors but got {len(inputs)}."
    input_indices = list(dict.fromkeys("".join(inputs)))  # ["a", "b", "c", "d", "e"]
    output_indices = list(dict.fromkeys(output))  # ["a", "c", "d", "e"]
    contraction_indices = [i for i in input_indices if i not in output_indices]  # ["b"]
    complexity_indices = contraction_indices + output_indices  # ["b", "a", "c", "d", "e"]
    symbols_multiplicities = {}  # {"M": 2, "N": 1, "K": 2}
    for i in complexity_indices:
        symbol = indices_to_symbols[i]
        if symbol in symbols_multiplicities:
            symbols_multiplicities[symbol] += 1
        else:
            symbols_multiplicities[symbol] = 1
    complexity_factors = []  # ["(M^2)", "(N)", "(K^2)"]
    for symbol, power in symbols_multiplicities.items():
        if power == 1:
            complexity_factors.append(f"({symbol})")
        else:
            complexity_factors.append(f"({symbol}^{power})")
    complexity = "".join(complexity_factors)  # "(M^2)(N)(K^2)"
    return complexity


def get_complexity(einsum_equation, indices_to_symbols, symbols_to_dimensions):
    """For a tensor contraction described by einsum_equation, compute the complexity equal to the
    sum of all single complexities of the optimal contraction path. The dimension symbol/value for 
    each index is given with the dictionary indices_to_symbols/symbols_to_dimensions. 

    example: (ABC)_adf = sum_bce A_abc B_bde C_cef
    einsum_equation = "abc,bde,cef->adf"
    indices_to_symbols = {"a": "M", "b": "M", "c": "N", "d": "K", "e": "K", "f": "N"}
    symbols_to_dimensions = {"M": 3, "N": 2, "K": 1}
    -> complexity of order (M^2)(N)(K^2) + (N^2)(K^2)(M)
    """
    input, _ = einsum_equation.split("->")  # "abc,bde,cef"
    tensors_indices = input.split(",")  # ["abc", "bde", "cef"]
    tensors = []  # [A_random, B_random, C_random]
    for tensor_indices in tensors_indices:
        shape = tuple([symbols_to_dimensions[indices_to_symbols[i]] for i in tensor_indices])
        tensor = np.random.normal(size=shape)
        tensors.append(tensor)
    _, path_info = oe.contract_path(einsum_equation, *tensors, optimize="optimal")
    total_complexity_terms = []  # ["(M^2)(N)(K^2)", "(N^2)(K^2)(M)"]
    for single_contraction in path_info.contraction_list:
        single_einsum_equation = single_contraction[2]
        single_complexity = get_single_complexity(single_einsum_equation, indices_to_symbols)
        total_complexity_terms.append(single_complexity)
    total_complexity = "+".join(total_complexity_terms)  # "(M^2)(N)(K^2)+(N^2)(K^2)(M)"
    return total_complexity