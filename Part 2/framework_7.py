import argparse
import numpy as np
import pandas as pd
import operator 

def framework(pairs, arr):
    """
    Args:
       - pairs:  a list of (cond, calc) tuples. calc() must be an executable
       - arr: a numpy array with the features in order feat_1, feat_2, ...
    
    Executes the first calc() whose cond returns True.
    Returns None if no condition matches.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
        
    return targets


def cond_eval(condition, arr):
    """evaluate a condition
        - condition: must be a tupe of (int, string, float). The second entry must be a string from the list below, describing the operator. Third entry of the tuple must be a float). If condition is None, it is always evaluated to true.
        - arr: array on which the condition is evaluated

    The python operator package is used. Second entry in condition must be one of those:
       ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    """
    ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True
    
    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


# TODO implement the missing parts of this function. You find an example below, main_example(args).
def main(args):
    """
    Main execution function for the Framework. 
    Loads data and defines the rules as sequential (condition, calculation) pairs.
    """
    # Load the full data array (expects feat_0 at index 0, feat_1 at index 1, etc.)
    data_array = pd.read_csv(args.eval_file_path).values

    # 1. Define Leaf Calculations (Linear Equations)
    # Using indices: 49 -> feat_49, 111 -> feat_111, 219 -> feat_219
    
    def leaf_5(r): 
        return -0.43 + (-0.15 * r[49]) + (-0.30 * r[219]) + (2.13 * r[111])
    
    def leaf_6(r): 
        return -0.28 + (-0.07 * r[49]) + (-0.49 * r[219]) + (1.72 * r[111])
    
    def leaf_7(r): 
        return 2.06 + (-12.38 * r[49]) + (-0.59 * r[219]) + (1.27 * r[111])
    
    def leaf_8(r): 
        return 10.18 + (-53.35 * r[49]) + (-0.66 * r[219]) + (0.71 * r[111])
    
    def leaf_11(r): 
        return -0.13 + (0.14 * r[49]) + (-0.94 * r[219]) + (-0.74 * r[111])
    
    def leaf_12(r): 
        return 0.07 + (0.53 * r[49]) + (-1.20 * r[219]) + (0.41 * r[111])
    
    def leaf_13(r): 
        return -0.03 + (-0.06 * r[49]) + (1.15 * r[219]) + (0.34 * r[111])
    
    def leaf_14(r): 
        return -0.04 + (-0.04 * r[49]) + (1.15 * r[219]) + (0.36 * r[111])

    # 2. Define Sequential Pair List
    # The framework executes the FIRST match. Logic flows from specific to general.
    pair_list = [
        # Path: feat_49 <= 0.22 
        # Inside this branch, we sub-split by feat_49 <= 0.17 and feat_111
        ((49, "<=", 0.17), lambda r: leaf_5(r) if r[111] <= 0.17 else leaf_6(r)),
        
        # This catch-all for (0.17 < feat_49 <= 0.22)
        ((49, "<=", 0.22), lambda r: leaf_7(r) if r[111] <= 0.41 else leaf_8(r)),
        
        # Path: feat_49 > 0.22
        ((49, "<=", 0.50), leaf_11),
        ((49, "<=", 0.71), leaf_12),
        ((49, "<=", 0.74), leaf_13),
        
        # Path: feat_49 > 0.74 (The final leaf)
        (None, leaf_14)
    ]
    
    return framework(pair_list, data_array)

def main_example(args):

    # Example: 
    test_arr = np.ones((10,10))

    def calc1(arr):
        """square first array column"""
        return arr[0]**2

    def calc2(arr):
        """add columns 3 and 4"""
        return arr[2] + arr[3]

    condition1 = (0,">=", 0.5)
    condition2 = (8, "==", 0.0)

    predict_targets = framework([(condition1, calc1), (condition2, calc2)], test_arr)
    print (predict_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_<ID>.csv")
    args = parser.parse_args()

    target02 = main(args)
    pd.DataFrame({"target02": target02}).to_csv("target02_predictions.csv", index=False)

