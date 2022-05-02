'''
Packages Used
'''
import numpy as np
import pandas as pd
import pulp
from pulp import *


'''
PORTFOLIO ALLOCATION
'''

def gen_bond(tenor, coupon, price):
    '''
    Generates a dictionary representing the cashflows of a bond  
    ---
    Parameters:  
        tenor (int): periods of interest until maturity of the bond  
        coupon (float): coupon payment of bond  
        price (float): price of the bond  
    ---
    Returns:  
        bond (dict): dictionary showing bond cashflows by period
    '''
    cfs = []
    cfs.append(-price)
    for i in range(1,tenor):
        cfs.append(coupon)
    cfs.append(100+coupon)
    bond = dict(zip(range(tenor+1), cfs))
    return bond

def bond_names(bond_count):
    '''
    Generates list of bond names as strings formatted for LaTeX in Markdown  
    ---
    Parameters:  
        bond_count (int): number of bonds in list
    ---
    Returns:  
        names (list): list of bonds names as strings formatted for LaTeX math-mode
    '''
    names = []
    for i in range(1,bond_count+1):
        names.append('$b_{' + str(i) + '}$')
    return names

def bond_data(tenors, prices, coupons):
    '''
    Generates a DataFrame of the bond cashflows over time  
    ---
    Parameters:  
        tenors (list): List of bond tenors (ints)  
        prices (list): List of bond priuces (floats)  
        coupons (list): List of bond coupons (floats)
    ---
    Returns:  
    df (DataFrame): DataFrame of bond cashflows
        x - cash flow years  
        y - bond names (formatted for LaTeX)
    '''
    import pandas as pd
    import numpy as np
    if len(tenors) == len(prices) == len(coupons):
        bond_count = len(tenors)
        bond_data = []
        for i in range(bond_count):
            bond_data.append(gen_bond(tenors[i], coupons[i], prices[i]))
        df = pd.DataFrame(bond_data, index=bond_names(bond_count)).fillna(0).transpose()
        df.index.name = 'cf_yr'
    else:
        print('ERROR: Tenors, Prices, and Coupons lists of different lengths')
    return df

def pv_factors(rates):
    '''
    Calculates Present Value Factors for given term structure of interest rates  
    ---
    Parameters:  
        rates (list): given term structure (float)  
    ---
    Returns:  
        pv (np array): PV Factors  
    '''
    import numpy as np
    pv = []
    for i in range(len(rates)):
        pv.append(1/(1+rates[i])**i)
    return np.asarray(pv)

def dur_factors(rates):
    '''
    Calculates Duration Factors for given term structure of interest rates  
    ---
    Parameters:  
        rates (list): given term structure (float)  
    ---
    Returns:  
        dur (np array): Duration Factors  
    '''
    import numpy as np
    dur = []
    for i in range(len(rates)):
        dur.append(i/(1+rates[i])**(i+1))
    return np.asarray(dur)

def conv_factors(rates):
    '''
    Calculates Convexity Factors for given term structure of interest rates  
    ---
    Parameters:  
        rates (list): given term structure (float)  
    ---
    Returns:  
        conv (np array): Convexity Factors  
    '''
    import numpy as np
    conv = []
    for i in range(len(rates)):
        conv.append((i*(i+1))/(1+rates[i])**(i+2))
    return np.asarray(conv)

def allocate_portfolio(liabilities, bonds_df, rates, r=0, immunization=False, duration=False, convexity=False, dedicated_years=0):
    '''
    Allocates Bond Portfolio by minimizing initial portfolio value subject to constraints (manipulatable)  
    ---
    Parameters:  
        liabilities (list): liability stream portfolio must cover (float)  
        bonds_df (DataFrame): bond data  
        r (float): interest rate on cash carry (default 0)  
        rates (list): current term structure of interest rates (float)  
        immunization (bool): solve the portfolio using present value immunized cashflows (default False)  
        duration (bool): solve the portfolio using duration immunized cashflows (default False)
        convexity (bool): adds convexity constraint to immunization solve (default False)  
        dedicated years (int): years dedicated to cashflow for immunized solve (default 0 in immunization)
    ---
    Returns:  
        Portfolio (LpProblem): Linear Program to be solved
    '''
    import pandas as pd
    import numpy as np
    import pulp

    '''Problem Type and Set Up'''
    bonds = bonds_df.columns
    if immunization:
        years = dedicated_years
        print(years)
    else:
        years = max(bonds_df.index) + 1

    '''Decision Variables'''
    bond_alloc = LpVariable.dicts('Bonds',bonds,lowBound=0)
    carry = LpVariable.dicts('CashCarry',range(max(bonds_df.index)+1),lowBound=0)

    '''Objective Function'''
    portfolio = LpProblem('Allocation',LpMinimize)
    portfolio += lpSum([bonds_df[i][0]*bond_alloc[i]*(-1) for i in bonds] + carry[0])
    
    '''Constraint - Dedication Years - Dedication & Immunization'''
    if years > 0:
        for i in range(1,years):
            portfolio += lpSum([bonds_df[j][i] * bond_alloc[j] for j in bonds] + carry[i-1]*(1+r) - carry[i]) >= liabilities[i]
    else:
        pass
    
    '''Constraint - NPV - Immunization'''
    if immunization:
        portfolio += lpSum([bonds_df[i][1:] * pv_factors(rates)[1:] * bond_alloc[i] for i in bonds]) == sum(liabilities * pv_factors(rates))
    else:
        pass

    '''Constraint - Duration - Immunization'''
    if duration:
        portfolio += lpSum([bonds_df[i][1:] * dur_factors(rates)[1:] * bond_alloc[i] for i in bonds]) == sum(liabilities * dur_factors(rates))
    else:
        pass

    '''Constraint - Convexity - Immunization'''
    if convexity:
        portfolio += lpSum([bonds_df[i][1:] * conv_factors(rates)[1:] * bond_alloc[i] for i in bonds]) == sum(liabilities * conv_factors(rates))
    else:
        pass

    return portfolio

def solve_portfolio(portfolio, bonds_df):
    '''
    Solves a Given PuLP Optimization Program and Outputs the Optimal Solution, Variable Allocation, and Sensitivity Information  
    ---
    Parameters:  
        problem (LpProblem) - Formulated LP   
        df (DataFrame) - Problem DataFrame  
    ---
    Returns:  
        final_dec (DataFrame) - Decision Variable Final Values  
        sp_df (DataFrame) - Shadow Prices  
    '''
    import pandas as pd
    import numpy as np
    import pulp

    '''Solves portfolio'''
    portfolio.solve()

    '''Decision Variable Values'''
    length = len(bonds_df.columns)
    var_alloc = dict(zip([v.name[6:] for v in portfolio.variables()[:length]], [v.varValue for v in portfolio.variables()[:length]]))    
    dec_vars = pd.DataFrame(var_alloc, index=['count']).transpose()
    
    '''Shadow Prices'''
    sp = [{'name':name, 'shadow_price':c.pi} for name, c in portfolio.constraints.items()]
    sp_df = pd.DataFrame(sp)

    '''Rediced Cost'''
    c = np.asarray(bonds_df.loc[0] * (-1))
    AT = bonds_df.transpose().iloc[:,1:].to_numpy()
    y = np.asarray(sp_df['shadow_price'])
    rc = np.round(c - np.matmul(AT,y),3)
    rc_df = pd.DataFrame(rc,index=bonds_df.columns)
    
    final_dec = pd.merge(dec_vars,rc_df, how='inner', left_index=True, right_index=True)
    final_dec.columns = ['var_val', 'reduced_cost']
    final_dec.index.name = 'Bond'
    
    return final_dec, sp_df

def derive_term_structure(sp_df):
    '''
    Derives implied term structure of interest rates given portfolio dedication shadow prices  
    ---
    Parameters:  
        sp_df (DataFrame): shadow price dataframe  
    ---
    Returns:  
        implied Rates (list): implied term structure of interest rates indexed from 0, list of floats
    '''
    import pandas as pd
    shadow_prices = sp_df['shadow_price'].to_list()
    implied_rates = [0]
    
    for i in range(len(shadow_prices)):
        implied_rates.append(1 / (shadow_prices[i] ** (1/(i+1))) - 1)

    return implied_rates


'''
Assorted Math Functions
'''

def downside_semi_variance(data):
    '''
    Returns the Downside Semi Variance of the Data Set  
    ---  
    Parameters:  
        data - (np array) Data array to calculate the dsv  
    ---  
    Returns:  
        dsv - (float) Downside Semi Variance
    '''
    import pandas as pd
    import numpy as np
    vals = np.where(data - np.mean(data) < 0, data - np.mean(data), 0)
    dsv = (1/len(data))*np.sum(np.square(vals))
    return dsv

def positive_semi_definite(A, print_result=True):
    '''
    Tests if the Matrix A is positive semidefinite  
    ---
    Parameters:  
        A - (matrix-like) Square matrix to be tests  
        print_result - (bool) Prints result for presentation (default True)
    ---
    Returns:  
        psd - (bool) True if PSD, else false [ONLY RETURNS WHEN print_result=False]
    '''
    import numpy.linalg as npl
    eigen = npl.eig(A)[0] >= 0
    psd = sum(eigen) == len(A)
    if print_result:
        if psd:
            print('The given matrix is Positive Semi Definite')
        else:
            print('The matrix is not Positive Semi Definite')
    else:
        return psd