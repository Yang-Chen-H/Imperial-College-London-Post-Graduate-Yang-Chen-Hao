import pandas as pd
import numpy as np
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD

# read and import data
def read_excel(input_file_path, sheet_name_input):
    df = pd.read_excel(input_file_path, sheet_name=sheet_name_input, header=None)
    df.columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH"]
    return df

# Check data accuracy (optional)
def write_excel(df, output_path, sheet_name_output):
    df.to_excel(output_path, sheet_name=sheet_name_output, index=False)

# define objective function and constraints using PuLP
def perform_optimization(data, q_values, p_values, n_values, i_values, max_carbon_tax, max_government_cost, market_shares, num_samples=10):
    results = []
    
    k_values = data.loc[17:528, 'K'].values * 1000  # Base value for green hydrogen production

    # Generate stochastic samples for green hydrogen price
    np.random.seed(0)  
    green_hydrogen_prices = np.random.normal(4.233, 1.05, num_samples)  

    # Define binary variables for each sample
    y_samples = [[LpVariable(f"y_{i}_{j}", cat="Binary") for i in range(512)] for j in range(num_samples)]
    M = 1e10  #  Big M method
    
    grey_price = 1.29

    for j in range(num_samples):
        # Create the problem
        prob = LpProblem("Maximize_Green_Market_Share", LpMaximize)

        # Define the decision variables
        renewable_electricity_tax_credit_rate = LpVariable("Renewable_Electricity_Tax_Credit_Rate", 0, None)
        H2_CFD_rate = LpVariable("H2_CFD_Rate", 0, None)
        carbon_tax_rate = LpVariable("Carbon_Tax_Rate", 0, None)

        price = green_hydrogen_prices[j]
        
        renewable_electricity_credits = q_values * renewable_electricity_tax_credit_rate
        H2_CFD = p_values * H2_CFD_rate
        carbon_tax_green = n_values * carbon_tax_rate
        carbon_tax_grey = i_values * carbon_tax_rate

        new_green_price_samples = k_values * price - renewable_electricity_credits - H2_CFD + carbon_tax_green
        new_grey_price_samples = k_values * grey_price + carbon_tax_grey

        for i in range(512):
            prob += new_green_price_samples[i] - new_grey_price_samples[i] <= M * (1 - y_samples[j][i])
            prob += new_grey_price_samples[i] - new_green_price_samples[i] <= M * y_samples[j][i]

        # Ensure the total carbon tax and total government cost do not exceed the specified limits
        total_carbon_tax = lpSum((n_values[i] if y_samples[j][i] else i_values[i]) * carbon_tax_rate for i in range(512))
        total_government_cost = lpSum(q_values * renewable_electricity_tax_credit_rate) + lpSum(p_values * H2_CFD_rate)
       
        prob += lpSum(market_shares[i] * y_samples[j][i] for i in range(512))  # Scale down the abs_diff term to balance the objective function
        prob += total_carbon_tax <= max_carbon_tax
        prob += total_government_cost <= max_government_cost
        
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
        optimal_params = [renewable_electricity_tax_credit_rate.varValue, H2_CFD_rate.varValue, carbon_tax_rate.varValue]
        switched_factories = [i + 18 for i in range(512) if y_samples[j][i].varValue == 1]

        total_carbon_tax_value = sum((n_values[i] if y_samples[j][i].varValue == 1 else i_values[i]) * optimal_params[2] for i in range(512))
        total_government_spend_value = sum(q_values * optimal_params[0]) + sum(p_values * optimal_params[1])

        results.append({
            "sample_index": j,
            "optimal_params": optimal_params,
            "max_market_share": value(prob.objective),
            "num_green_factories": sum(value(y_samples[j][i]) == 1 for i in range(512)),
            "switched_factories": switched_factories,
            "total_carbon_tax": total_carbon_tax_value,
            "total_government_spend": total_government_spend_value,
            "green_hydrogen_price": price
        })
        
        # If all factories switched, find minimal policy support
        if sum(value(y_samples[j][i]) == 1 for i in range(512)) == 512:
            prob_min_support = LpProblem("Minimize_Policy_Support", LpMinimize)

            # Add decision variables
            renewable_electricity_tax_credit_rate_min = LpVariable("Renewable_Electricity_Tax_Credit_Rate_Min", 0, None)
            H2_CFD_rate_min = LpVariable("H2_CFD_Rate_Min", 0, None)
            carbon_tax_rate_min = LpVariable("Carbon_Tax_Rate_Min", 0, None)
            
            # Recalculate costs with new decision variables
            renewable_electricity_credits_min = q_values * renewable_electricity_tax_credit_rate_min
            H2_CFD_min = p_values * H2_CFD_rate_min
            carbon_tax_green_min = n_values * carbon_tax_rate_min
            carbon_tax_grey_min = i_values * carbon_tax_rate_min

            new_green_price_samples_min = k_values * price - renewable_electricity_credits_min - H2_CFD_min + carbon_tax_green_min
            new_grey_price_samples_min = k_values * grey_price + carbon_tax_grey_min
            
            total_carbon_tax_min = lpSum((n_values[i] if y_samples[j][i] else i_values[i]) * carbon_tax_rate_min for i in range(512))
            total_government_cost_min = lpSum(q_values * renewable_electricity_tax_credit_rate_min) + lpSum(p_values * H2_CFD_rate_min)

            for i in range(512):
                prob_min_support += new_green_price_samples_min[i] - new_grey_price_samples_min[i] <= 0

            # Add constraints to ensure all factories stay switched
            for i in range(512):
                prob_min_support += y_samples[j][i] == 1

            # Minimize the sum of policy support rates
            prob_min_support += renewable_electricity_tax_credit_rate_min * (78000) + H2_CFD_rate_min + 2 * carbon_tax_rate_min
            
            prob_min_support += total_carbon_tax_min <= max_carbon_tax
            prob_min_support += total_government_cost_min <= max_government_cost
            
            prob_min_support.solve(solver)

            minimal_support_params = [renewable_electricity_tax_credit_rate_min.varValue, H2_CFD_rate_min.varValue, carbon_tax_rate_min.varValue]

            # Update results with minimal support policies
            results[-1]["optimal_params"] = minimal_support_params
            results[-1]["total_carbon_tax"] = sum((n_values[i] if y_samples[j][i].varValue == 1 else i_values[i]) * minimal_support_params[2] for i in range(512))
            results[-1]["total_government_spend"] = sum(q_values * minimal_support_params[0]) + sum(p_values * minimal_support_params[1])

    return results

# main function
def main():
    input_file_path = 'C:/graduate project models/green hydrogen/hydrogen_refinery_data.xlsx'
    output_path = 'C:/graduate project models/green hydrogen/output.xlsx'
    output_txt_path = 'C:/graduate project models/green hydrogen/output.txt'
    sheet_name_input = 'model'
    sheet_name_output = 'Sheet1'

    # read data
    data = read_excel(input_file_path, sheet_name_input)

    q_values = data.loc[17:528, "Q"].values
    p_values = data.loc[17:528, "P"].values
    n_values = data.loc[17:528, 'N'].values
    i_values = data.loc[17:528, 'I'].values
    market_shares = data.loc[17:528, 'J'].values

    write_excel(data, output_path, sheet_name_output)

    # Define the maximum allowable costs
    max_carbon_tax = 3e10
    max_government_cost = 5e10

    # perform optimization
    results = perform_optimization(data, q_values, p_values, n_values, i_values, max_carbon_tax, max_government_cost, market_shares)

    with open(output_txt_path, 'w') as f:
        for result in results:
            f.write(f"Sample Index: {result['sample_index']}\n")
            f.write("Optimal Renewable Electricity Tax Credit: {:.6f}\n".format(result["optimal_params"][0]))
            f.write("Optimal H2 CFD Rate: {:.6f}\n".format(result["optimal_params"][1]))
            f.write("Optimal Carbon Tax Rate: {:.6f}\n".format(result["optimal_params"][2]))
            f.write("Maximum Green Market Share: {:.6f}\n".format(result["max_market_share"]))
            f.write("Number of factories switching to green: {}\n".format(result["num_green_factories"]))
            f.write("Factories that switched to green: {}\n".format(result["switched_factories"]))
            f.write("Total Carbon Tax: {:.6f}\n".format(result["total_carbon_tax"]))
            f.write("Total Government Spend: {:.6f}\n".format(result["total_government_spend"]))
            f.write("Green Hydrogen Price: {:.6f}\n".format(result["green_hydrogen_price"]))
            f.write("\n")

    for result in results:
        print(f"Sample Index: {result['sample_index']}")
        print("Optimal Renewable Electricity Tax Credit:", result["optimal_params"][0])
        print("Optimal H2 CFD Rate:", result["optimal_params"][1])
        print("Optimal Carbon Tax Rate:", result["optimal_params"][2])
        print("Maximum Green Market Share:", result["max_market_share"])
        print("Number of factories switching to green:", result["num_green_factories"])
        print("Factories that switched to green:", result["switched_factories"])
        print("Total Carbon Tax:", result["total_carbon_tax"])
        print("Total Government Spend:", result["total_government_spend"])
        print("Green Hydrogen Price (Sample):", result["green_hydrogen_price"])
        print("\n")

if __name__ == "__main__":
    main()
