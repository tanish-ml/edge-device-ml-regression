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


def main(args):

    # Load data (feat_22=idx 22, feat_49=idx 49, feat_111=idx 111, feat_219=idx 219)
    data_array = pd.read_csv(args.eval_file_path).values

    # 1. Define Calculations (Polynomial Equations)

    def calc_1(r): 
        return (-0.45 * r[219]) + (1.75 * r[111]) + (-0.65 * r[22])

    def calc_2(r): 
        return (-51.95 + (544.10 * r[49]) + (1.86 * r[219]) + (14.76 * r[111]) + (-2.67 * r[22]) + 
                (-1418.28 * r[49]**2) + (-14.08 * r[49] * r[219]) + (-70.67 * r[49] * r[111]) + 
                (11.08 * r[49] * r[22]) + (0.23 * r[219]**2) + (-0.34 * r[219] * r[111]) + 
                (0.26 * r[219] * r[22]) + (0.06 * r[111]**2) + (-0.27 * r[111] * r[22]) + (0.07 * r[22]**2))

    def calc_3(r): 
        return (-0.95 * r[219]) + (-0.75 * r[111]) + (-0.15 * r[22])

    def calc_4(r): 
        return (152.89 + (-637.30 * r[49]) + (-3.28 * r[219]) + (-4.60 * r[111]) + (2.04 * r[22]) + 
                (663.22 * r[49]**2) + (5.08 * r[49] * r[219]) + (8.74 * r[49] * r[111]) + 
                (-4.14 * r[49] * r[22]) + (-0.02 * r[219]**2) + (-0.14 * r[219] * r[111]) + 
                (-0.02 * r[219] * r[22]) + (-0.08 * r[111]**2) + (-0.29 * r[111] * r[22]) + (-0.07 * r[22]**2))

    def calc_5(r): 
        return (-1.25 * r[219]) + (0.45 * r[111]) + (0.75 * r[22])

    def calc_6(r): 
        return (499.29 + (-1457.31 * r[49]) + (-32.97 * r[219]) + (3.36 * r[111]) + (12.42 * r[22]) + 
                (1062.88 * r[49]**2) + (47.23 * r[49] * r[219]) + (-4.07 * r[49] * r[111]) + 
                (-16.99 * r[49] * r[22]) + (-0.21 * r[219]**2) + (0.14 * r[219] * r[111]) + 
                (-0.30 * r[219] * r[22]) + (-0.18 * r[111]**2) + (-0.13 * r[111] * r[22]) + (0.03 * r[22]**2))

    def calc_7(r): 
        return (1.15 * r[219]) + (0.35 * r[111]) + (-0.15 * r[22])

    # 2. Define Sequential Pair List
    # The framework executes the FIRST match. The logic follows the feat_49 splits.
    pair_list = [
        ((49, "<=", 0.17), calc_1),  # 0.1701
        ((49, "<=", 0.21), calc_2),  # 0.2147
        ((49, "<=", 0.46), calc_3),  # 0.4615
        ((49, "<=", 0.50), calc_4),  # 0.5023 (End of Left Tree)
        ((49, "<=", 0.67), calc_5),  # 0.6653
        ((49, "<=", 0.71), calc_6),  # 0.7059
        ((49, ">", 0.71), calc_7),  # 0.7438
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

