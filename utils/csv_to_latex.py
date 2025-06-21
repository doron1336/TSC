import pandas as pd

# Read the CSV file
df = pd.read_csv('/Users/doron/Desktop/personal/thesis/TSC/UCRArchive_2018/scores_dataframe.csv')
df_without_description = df.iloc[:, :-1]


# Format string columns with \texttt and float columns to 4 decimal places
def format_value(x):
    if isinstance(x, str):
        return f"\\texttt{{{x}}}"
    elif isinstance(x, float):
        return f"{x:.4f}"  # Format to 4 decimal places
    else:
        return x


df_tex = df_without_description.applymap(format_value)

# Creating a longtable with grid lines
with open('table.tex', 'w') as f:
    # Write longtable header
    f.write("\\begin{longtable}{|" + "|".join(["l"] * 2 + ["r"] * (len(df_tex.columns) - 2)) + "|}\n")
    f.write("\\hline\n")

    # Write column headers
    header_row = " & ".join([f"{col}" for col in df_tex.columns]) + " \\\\\n"
    f.write(header_row)
    f.write("\\hline\n")

    # Add header configuration for multi-page tables
    f.write("\\endhead\n")
    f.write("\\hline\n")
    f.write("\\multicolumn{" + str(len(df_tex.columns)) + "}{|r|}{Continued on next page} \\\\\n")
    f.write("\\hline\n")
    f.write("\\endfoot\n")
    f.write("\\hline\n")
    f.write("\\endlastfoot\n")

    # Write data rows
    for _, row in df_tex.iterrows():
        # Check if the Type column (assuming it's the second column, index 1) is HAR
        if row[15] == "\\texttt{HAR}":  # After formatting, HAR is wrapped in \texttt
            f.write("\\rowcolor[gray]{0.9} ")  # Light gray background for entire HAR row
        row_str = " & ".join([str(val) for val in row.values]) + " \\\\\n"
        f.write(row_str)
        f.write("\\hline\n")

    # End the longtable
    f.write("\\end{longtable}\n")