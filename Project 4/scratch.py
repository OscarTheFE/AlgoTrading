import pandas as pd
if __name__ == "__main__":
    w_b = pd.Series(index=["a","b"], data=1)
    w_b.USDOLLAR = 0.  # add risk free cash account
    w_b /= sum(w_b)  # normalize the weight and sum to 1
    print(1E8)