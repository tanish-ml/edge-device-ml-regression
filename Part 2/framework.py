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
    pair_list = [] # TODO implement pairs of (condition, calculation), see example below
    
    data_array = pd.read_csv(args.eval_file_path).values
    
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
    main_example(args)
    target02 = main(args)
