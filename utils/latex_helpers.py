import pytablewriter
import pandas as pd 


def df_to_latex(df: pd.DataFrame):
	writer = pytablewriter.LatexTableWriter()
	writer.from_dataframe(df)
	writer.write_table()

	return

