import pandas as pd
import numpy as np
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD

# read and import data
def read_excel(input_file_path, sheet_name_input):
    df = pd.read_excel(input_file_path, sheet_name=sheet_name_input, header=None)
    df.columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ","AK","AL"]
    return df

# Check data accuracy （optional）
def write_excel(df, output_path, sheet_name_output):
    df.to_excel(output_path, sheet_name=sheet_name_output, index=False)

# define objective function and constraints using PuLP
def perform_optimization(data, p_values, n_values, i_values, market_shares, num_samples=10):
    results = []
    
    k_values = data.loc[17:528, 'K'].values * 1000  # Base value for blue hydrogen production

    # Generate stochastic samples for blue hydrogen price
    np.random.seed(0)  # For reproducibility
    blue_hydrogen_prices = np.random.normal(1.95, 0.2362, num_samples)  

    # Define binary variables for each sample
    y_samples = [[LpVariable(f"y_{i}_{j}", cat="Binary") for i in range(512)] for j in range(num_samples)]
    M = 1e11  #  Big M method
    
    grey_price = 1.29

    for j in range(num_samples):
        # Create the problem
        prob = LpProblem("Optimize_Policy", LpMaximize)

        # Define the decision variables
        
        H2_CFD_rate = LpVariable("H2_CFD_Rate", 0, 300)
        renewable_hydrogen_tax_credit_rate = LpVariable("Renewable_Hydrogen_Tax_Credit_Rate", 0, 1)
        carbon_tax_rate = LpVariable("Carbon_Tax_Rate", 0, 132)

        price = blue_hydrogen_prices[j]
        
        
        H2_CFD = p_values * H2_CFD_rate
        renewable_hydrogen_credits = k_values  * renewable_hydrogen_tax_credit_rate
        carbon_tax_blue = n_values * carbon_tax_rate
        carbon_tax_grey = i_values * carbon_tax_rate

        new_blue_price_samples = k_values * price - H2_CFD - renewable_hydrogen_credits + carbon_tax_blue
        new_grey_price_samples = k_values * grey_price + carbon_tax_grey

        for i in range(512):
            prob += new_blue_price_samples[i] - new_grey_price_samples[i] <= M * (1 - y_samples[j][i])
            prob += new_grey_price_samples[i] - new_blue_price_samples[i] <= M * y_samples[j][i]

        

        

        # Objective function to maximize blue market share 
        prob += lpSum(market_shares[i] * y_samples[j][i] for i in range(512)) 
       
      
        # Solve the problem
        solver = PULP_CBC_CMD(msg=True)
        prob.solve(solver)

        # obtain the number of parameters in the model
        objective_parameters_count = sum(1 for v in prob.objective.keys())
        
        constraint_parameters_count = sum(sum(1 for v in constraint.keys()) for constraint in prob.constraints.values())
        total_parameters_count = objective_parameters_count + constraint_parameters_count
        print(f"total_number_of_parameters: {total_parameters_count}")
        
        print(f"total_objective_parameters: {objective_parameters_count}")
        
        # obtain the number of variables and constraints in the model
        num_vars = len(prob.variables())
        num_constraints = len(prob.constraints)
        

        print(f"\nnumber_of_variables_in_the_model: {num_vars}")
        print(f"number_of_constraints_in_the_model: {num_constraints}")
        

        # Extract results
        optimal_params = [H2_CFD_rate.varValue, renewable_hydrogen_tax_credit_rate.varValue, carbon_tax_rate.varValue]
        switched_factories = [i + 18 for i in range(512) if y_samples[j][i].varValue == 1]

        total_carbon_tax_value = sum((n_values[i] if y_samples[j][i].varValue == 1 else i_values[i]) * optimal_params[2] for i in range(512))
        total_government_cost_value = sum(p_values * optimal_params[0]) + sum(k_values  * optimal_params[1])

        results.append({
            "sample_index": j,
            "optimal_params": optimal_params,
            "max_market_share": value(lpSum(market_shares[i] * y_samples[j][i] for i in range(512))),
            "num_blue_factories": sum(value(y_samples[j][i]) == 1 for i in range(512)),
            "switched_factories": switched_factories,
            "total_carbon_tax": total_carbon_tax_value,
            "total_government_cost": total_government_cost_value,
            "blue_hydrogen_price": price
        })
        
        # If all factories switched, find minimal policy support
        if sum(value(y_samples[j][i]) == 1 for i in range(512)) == 512:
            prob_min_support = LpProblem("Minimize_Policy_Support", LpMinimize)

            # Add decision variables
            H2_CFD_rate_min = LpVariable("H2_CFD_Rate", 0, 300)
            renewable_hydrogen_tax_credit_rate_min = LpVariable("Renewable_Hydrogen_Tax_Credit_Rate", 0, 1)
            carbon_tax_rate_min = LpVariable("Carbon_Tax_Rate", 0, 132)
            
            # Recalculate costs with new decision variables
            H2_CFD_min = p_values * H2_CFD_rate_min
            renewable_hydrogen_credits_min = k_values  * renewable_hydrogen_tax_credit_rate_min
            carbon_tax_blue_min = n_values * carbon_tax_rate_min
            carbon_tax_grey_min = i_values * carbon_tax_rate_min

            new_blue_price_samples_min = k_values * price - H2_CFD_min - renewable_hydrogen_credits_min + carbon_tax_blue_min
            new_grey_price_samples_min = k_values * grey_price + carbon_tax_grey_min

            for i in range(512):
                prob_min_support += new_blue_price_samples_min[i] - new_grey_price_samples_min[i] <= 0

            # Add constraints to ensure all factories stay switched
            for i in range(512):
                prob_min_support += y_samples[j][i] == 1

            # Minimize the sum of policy support rates
            prob_min_support += renewable_hydrogen_tax_credit_rate_min * (500) + H2_CFD_rate_min + 2 * carbon_tax_rate_min
            
            prob_min_support.solve(solver)

            minimal_support_params = [ H2_CFD_rate_min.varValue, renewable_hydrogen_tax_credit_rate_min.varValue, carbon_tax_rate_min.varValue]

            # Update results with minimal support policies
            results[-1]["optimal_params"] = minimal_support_params
            results[-1]["total_carbon_tax"] = sum((n_values[i] if y_samples[j][i].varValue == 1 else i_values[i]) * minimal_support_params[2] for i in range(512))
            results[-1]["total_government_spend"] = sum(p_values * minimal_support_params[0]) + sum(k_values  * minimal_support_params[1])

    return results

# Main function
def main():
    input_file_path = 'C:/graduate project models/blue hydrogen/blue_hydrogen_data.xlsx'
    
    output_txt_path = 'C:/graduate project models/blue hydrogen/output.txt'
    sheet_name_input = 'model'

    # Read data 
    data = read_excel(input_file_path, sheet_name_input)

    

    p_values = data.loc[17:528, "P"].values
    n_values = data.loc[17:528, 'N'].values
    i_values = data.loc[17:528, 'I'].values
    market_shares = data.loc[17:528, 'J'].values


    # Perform optimization
    results = perform_optimization(data, p_values, n_values, i_values, market_shares)

    with open(output_txt_path, 'w') as f:
        for result in results:
            f.write(f"Sample Index: {result['sample_index']}\n")
            f.write("Optimal H2 CFD Rate: {:.6f}\n".format(result["optimal_params"][0]))
            f.write("Optimal Renewable Hydrogen Tax Credit Rate: {:.6f}\n".format(result["optimal_params"][1]))
            f.write("Optimal Carbon Tax Rate: {:.6f}\n".format(result["optimal_params"][2]))
            f.write("Maximum Blue Market Share: {:.6f}\n".format(result["max_market_share"]))
            f.write("Number of factories switching to blue: {}\n".format(result["num_blue_factories"]))
            f.write("Factories that switched to blue: {}\n".format(result["switched_factories"]))
            f.write("Total Carbon Tax (borne by companies): {:.6f}\n".format(result["total_carbon_tax"]))
            f.write("Total Government Budget: {:.6f}\n".format(result["total_government_cost"]))
            f.write("Blue Hydrogen Price (Sample): {:.6f}\n".format(result["blue_hydrogen_price"]))
            f.write("\n")

    for result in results:
        print(f"Sample Index: {result['sample_index']}")
        print("Optimal H2 CFD Rate:", result["optimal_params"][0])
        print("Optimal Renewable Hydrogen Tax Credit Rate:", result["optimal_params"][1])
        print("Optimal Carbon Tax Rate:", result["optimal_params"][2])
        print("Maximum Blue Market Share:", result["max_market_share"])
        print("Number of factories switching to blue:", result["num_blue_factories"])
        print("Factories that switched to blue:", result["switched_factories"])
        print("Total Carbon Tax (borne by companies):", result["total_carbon_tax"])
        print("Total Government Cost Budget:", result["total_government_cost"])
        print("Blue Hydrogen Price (Sample):", result["blue_hydrogen_price"])
        print("\n")

if __name__ == "__main__":
    main()
