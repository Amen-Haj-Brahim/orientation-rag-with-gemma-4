import tabula
# Converts all tables in the PDF to a single CSV
tabula.convert_into("sd_par_dom_23_24_25.pdf", "sd_par_dom_23_24_25.csv", output_format="csv", pages='all')